# -*- coding: UTF-8 -*-

import functools
import hasher

import numpy as np
import scipy.spatial.distance as distance

from pyspark.mllib.linalg import SparseVector


def distance_metric(kv):
    """
    Generates a pairwise, summed Jaccard distance metric for all the elements
    in a cluster/bucket. 
    Returns a <k, v> pair: <bucket id, jaccard distance>.

    Parameters
    ----------
    kv: tuple of (int, list of array_like)
        A tuple of the form <k, v> pair where k is the cluster/bucket
        id and v a list of vectors.

    Returns
    -------
    bid: int
        The cluster/bucket id
    distance: float
        The average Jaccard distance of the elements of the bucket.
    """
    bid, X = kv[0], kv[1].data
    if type(X[0]) is SparseVector:
        X = np.array([x.toArray() for x in X])
    
    return [bid, distance.pdist(np.array(X), 'jaccard').mean()]


class PyLSHModel:
    """
    Wrapper class for LSH model.
    """
    def __init__(self, budget, min_clusters=2, target_threshold=None, n_bands=None):
        """
        Initialize the LSH model. Only one of the parameters target_threshold or
        n_bands is required.

        Parameters
        ----------
        budget : integer
            Total number of rows to split the signatures into.
        min_clusters : integer
            Minimum allowable cluster size.
        target_threshold: float, optional
            Value of desired threshold if bands not specified.
        n_bands : integer, optional
            Number of bands.
        """
        
        self.budget = budget # budget is the total number of rows: rows*bands
        self.target_threshold = target_threshold
        self.min_clusters = min_clusters
        if n_bands:
            self.n_bands = n_bands
            self.n_rows = budget / n_bands
            self.threshold = target_threshold
        else:
            self.__tune_parameters()
                
        self.sigs = None
        self.bands = None
        self.vectors_buckets = None
        self.buckets_vectors = None
        self.buckets = None
        self.scores = None
        
    def __tune_parameters(self):
        for bands in xrange(1, self.budget / 2):
            if self.budget % bands == 0:
                rows = self.budget / bands
                threshold = (1.0 / bands) ** (1.0 / rows)
                if (threshold < self.target_threshold):
                    self.n_bands = bands
                    self.n_rows = rows
                    self.threshold = threshold
                    return
                
    def run(self, data, p, m):
        """
        Starts the main LSH process.

        Parameters
        ----------
        data : RDD[Vector]
            RDD of data points. Acceptable vector types are numpy.ndarray,
            list or PySpark SparseVector.
        p : integer
            Prime number larger than the largest value in data.
        m : integer
            Number of bins for hashing.
        """

        zdata = data.zipWithIndex()

        seeds = np.vstack([np.random.random_integers(p, size=self.budget), np.random.random_integers(0, p, size=self.budget)]).T
        hashes = [functools.partial(hasher.minhash, a=s[0], b=s[1], p=p, m=m) for s in seeds]

        # Start by generating the signatures for each data point.
        # Output format is:
        # <(vector idx, band idx), (row idx, minhash)>
        sigs = zdata.flatMap(lambda x: [[(x[1], i % self.n_bands), (i, h(x[0]))] for i, h in enumerate(hashes)]).cache()

        # Put together the vector minhashes in the same band.
        # Output format is:
        # <(band idx, hash minhash-list), vector idx>
        bands = sigs.groupByKey().mapValues(sorted) \
            .map(lambda x: [(x[0][1], hash(tuple(x[1]))), x[0][0]]) \
            .groupByKey().cache()

        # Filter the bucket with size < min_clusters
        if self.min_clusters > 0:
            bands = bands.filter(lambda x: len(x[1]) >= self.min_clusters).cache()

        # Remaps each element to a cluster / bucket index.
        # Output format is:
        # <vector idx, bucket idx>
        vector_bucket = bands.map(lambda x: frozenset(sorted(x[1]))).distinct() \
            .zipWithIndex().flatMap(lambda x: map(lambda y: (np.long(y), x[1]), x[0])) \
            .cache()

        # Reverses indices, to key the vectors by their buckets.
        # Output format is:
        # <bucket idx, vector idx>
        bucket_vector = vector_bucket.map(lambda x: (x[1], x[0])).cache()

        # Joins indices up with original data to provide clustering results.
        # Output format is:
        # <bucket idx, list of vectors>
        buckets = zdata.map(lambda x: (x[1], x[0])).join(vector_bucket) \
            .map(lambda x: (x[1][1], x[1][0])).groupByKey().cache()

        # Computes Jaccard similarity of each bucket.
        scores = buckets.map(distance_metric).cache()
        
        # Update the class fields at the end to avoid inconsistencies
        self.signatures = sigs
        self.bands = bands
        self.vectors_buckets = vector_bucket
        self.buckets_vectors = bucket_vector
        self.buckets = buckets
        self.scores = scores
