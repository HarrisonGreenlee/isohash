#pragma once
#include <vector>
#include <armadillo>

// // Function to retrieve the combined node hashes of two matrices
// std::pair<arma::Col<long long>, arma::Col<long long>> nodeHashCompare(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n);

// // Function to retrieve the combined edge hashes of two matrices
// std::pair<arma::Col<long long>, arma::Col<long long>> edgeHashCompare(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n);

// bool testNodeHashEquality(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n);

// bool testEdgeHashEquality(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n);
bool nodeHashCompare(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n);
bool edgeHashCompare(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n);