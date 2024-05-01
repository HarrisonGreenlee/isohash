#include <iostream>
#include <vector>
#include <armadillo>
#include <algorithm>
#include <functional>
#include <string>

// Function to hash columns of a matrix, after sorting the elements in each column
std::vector<size_t> columnHashSorted(const arma::Mat<long long>& matrix) {
    std::vector<size_t> hashes;
    std::hash<std::string> hasher;

    // Hash each column after sorting
    for (size_t col = 0; col < matrix.n_cols; ++col) {
        std::vector<int> columnData(matrix.n_rows);
        for (size_t row = 0; row < matrix.n_rows; ++row) {
            columnData[row] = matrix(row, col);
        }
        std::sort(columnData.begin(), columnData.end());  // Sort the column before hashing
        std::string colStr;
        for (int value : columnData) {
            colStr += std::to_string(value) + ",";
        }
        size_t hashValue = hasher(colStr);
        hashes.push_back(hashValue);
    }

    return hashes;
}

// Function to perform matrix multiplication, hashing, and storing results separately for each matrix
std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<size_t>>>
nodeHeuristic(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n) {
    std::vector<std::vector<size_t>> hashesA;
    std::vector<std::vector<size_t>> hashesB;

    arma::Mat<double> currentA = arma::conv_to<arma::Mat<double>>::from(matrixA);
    arma::Mat<double> currentB = arma::conv_to<arma::Mat<double>>::from(matrixB);

    for (int i = 0; i < n; ++i) {
        // Perform matrix multiplication
        currentA = currentA * currentA;
        currentB = currentB * currentB;

        // Round the result of the multiplication to the nearest integer and update current matrices
        currentA = arma::round(currentA);
        currentB = arma::round(currentB);

        // Convert rounded matrices to long long type
        arma::Mat<long long> longA = arma::conv_to<arma::Mat<long long>>::from(currentA);
        arma::Mat<long long> longB = arma::conv_to<arma::Mat<long long>>::from(currentB);

        // Hash the columns of each matrix
        std::vector<size_t> hashA = columnHashSorted(longA);
        std::vector<size_t> hashB = columnHashSorted(longB);

        // Store the hashes of this iteration
        hashesA.push_back(hashA);
        hashesB.push_back(hashB);

        // Compare hashes
        std::vector<size_t> sortedHashA = hashA;
        std::vector<size_t> sortedHashB = hashB;
        std::sort(sortedHashA.begin(), sortedHashA.end());
        std::sort(sortedHashB.begin(), sortedHashB.end());

        if (sortedHashA != sortedHashB) {
            // Hash histograms differ, which is sufficient to prove that graph isomorphism between A and B is impossible
            return {std::vector<std::vector<size_t>>{hashA}, std::vector<std::vector<size_t>>{hashB}};
        }
    }

    return {hashesA, hashesB};
}

// Combine node hashes from multiple matrices into a single hash per column
std::vector<size_t> combinedNodeHeuristic(const std::vector<std::vector<size_t>>& hashResults) {
    std::vector<size_t> combinedHashes;
    std::hash<std::string> hasher;

    // Determine the number of columns (nodes)
    if (hashResults.empty()) return combinedHashes;
    int numColumns = hashResults[0].size();

    // Combine hashes column-wise across all path lengths
    for (int col = 0; col < numColumns; ++col) {
        std::string combinedStr;
        for (const auto& row : hashResults) {
            combinedStr += std::to_string(row[col]) + ",";
        }
        size_t combinedHashValue = hasher(combinedStr);
        combinedHashes.push_back(combinedHashValue);
    }

    return combinedHashes;
}

// Function to compute path-based identifiers for edges for various powers of the matrix
std::pair<std::vector<std::vector<size_t>>, std::vector<std::vector<size_t>>> edgeHeuristic(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n) {
    std::vector<std::vector<size_t>> edgeHashesA;
    std::vector<std::vector<size_t>> edgeHashesB;
    std::hash<std::string> hasher;

    arma::Mat<long long> currentA = matrixA;
    arma::Mat<long long> currentB = matrixB;

    for (int i = 0; i < n; ++i) {
        std::vector<size_t> hashesA;
        std::vector<size_t> hashesB;

        for (size_t row = 0; row < currentA.n_rows; ++row) {
            for (size_t col = 0; col < currentA.n_cols; ++col) {
                std::string hashStrA = std::to_string(currentA(row, col));
                std::string hashStrB = std::to_string(currentB(row, col));
                size_t hashA = hasher(hashStrA);
                size_t hashB = hasher(hashStrB);
                hashesA.push_back(hashA);
                hashesB.push_back(hashB);
            }
        }

        edgeHashesA.push_back(hashesA);
        edgeHashesB.push_back(hashesB);

        currentA = matrixA * currentA; // Multiply matrices to get paths of length i+1
        currentB = matrixB * currentB;

        std::vector<size_t> sortedHashesA = hashesA;
        std::vector<size_t> sortedHashesB = hashesB;
        std::sort(sortedHashesA.begin(), sortedHashesA.end());
        std::sort(sortedHashesB.begin(), sortedHashesB.end());

        if (sortedHashesA != sortedHashesB) {
            return {std::vector<std::vector<size_t>>{hashesA}, std::vector<std::vector<size_t>>{hashesB}};
        }
    }

    return {edgeHashesA, edgeHashesB};
}

// Function to combine edge hashes from multiple matrix powers into a single hash per edge
std::vector<size_t> combinedEdgeHeuristic(const std::vector<std::vector<size_t>>& edgeHashResults) {
    std::vector<size_t> combinedEdgeHashes;
    std::hash<std::string> hasher;

    if (edgeHashResults.empty()) return combinedEdgeHashes;

    int numRows = sqrt(edgeHashResults[0].size());  // Since edges are stored linearly for each power, compute dimension assuming square matrices
    int numColumns = numRows;

    // Combine hashes edge-wise across all path lengths
    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numColumns; ++col) {
            std::string combinedStr;
            for (const auto& hashes : edgeHashResults) {
                combinedStr += std::to_string(hashes[row * numColumns + col]) + ",";
            }
            size_t combinedHashValue = hasher(combinedStr);
            combinedEdgeHashes.push_back(combinedHashValue);
        }
    }

    return combinedEdgeHashes;
}

// Function to compare the combined node hashes of two matrices
bool nodeHashCompare(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n) {
    auto [resultA, resultB] = nodeHeuristic(matrixA, matrixB, n);
    auto combinedHashesA = combinedNodeHeuristic(resultA);
    auto combinedHashesB = combinedNodeHeuristic(resultB);
    std::sort(combinedHashesA.begin(), combinedHashesA.end());
    std::sort(combinedHashesB.begin(), combinedHashesB.end());
    return combinedHashesA == combinedHashesB;
}

// Function to compare the combined edge hashes of two matrices
bool edgeHashCompare(const arma::Mat<long long>& matrixA, const arma::Mat<long long>& matrixB, int n) {
    auto [resultA, resultB] = edgeHeuristic(matrixA, matrixB, n);
    auto combinedHashesA = combinedEdgeHeuristic(resultA);
    auto combinedHashesB = combinedEdgeHeuristic(resultB);
    std::sort(combinedHashesA.begin(), combinedHashesA.end());
    std::sort(combinedHashesB.begin(), combinedHashesB.end());
    return combinedHashesA == combinedHashesB;
}

// int main() {
//     arma::Mat<long long> matrixA = {
//         {0, 1, 1},
//         {1, 0, 0},
//         {1, 0, 0}
//     };

//     arma::Mat<long long> matrixB = {
//         {0, 0, 1},
//         {0, 0, 1},
//         {1, 1, 0}
//     };

//     auto [resultA, resultB] = nodeHeuristic(matrixA, matrixB, 3);

//     std::cout << "Column Hashes for Matrix A:" << std::endl;
//     for (const auto& hashes : resultA) {
//         for (size_t hash : hashes) {
//             std::cout << "Hash: " << hash << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "Column Hashes for Matrix B:" << std::endl;
//     for (const auto& hashes : resultB) {
//         for (size_t hash : hashes) {
//             std::cout << "Hash: " << hash << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "Combined Node Hashes for Matrix A" << std::endl;
//     auto combinedHashesA = combinedNodeHeuristic(resultA);
//     for (size_t hash : combinedHashesA) {
//         std::cout << "Hash: " << hash << std::endl;
//     }

//     std::cout << std::endl;

//     std::cout << "Combined Node Hashes for Matrix B" << std::endl;
//     auto combinedHashesB = combinedNodeHeuristic(resultB);
//     for (size_t hash : combinedHashesB) {
//         std::cout << "Hash: " << hash << std::endl;
//     }

//     auto [hashesA, hashesB] = edgeHeuristic(matrixA, matrixB, 3);

//     std::cout << std::endl;

//     std::cout << "Edge hashes for Matrix A:" << std::endl;
//     for (const auto& iteration : hashesA) {
//         for (const size_t hash : iteration) {
//             std::cout << "Hash: " << hash << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "Edge hashes for Matrix B:" << std::endl;
//     for (const auto& iteration : hashesB) {
//         for (const size_t hash : iteration) {
//             std::cout << "Hash: " << hash << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     std::cout << std::endl;

//     auto combinedEdgeHashesA = combinedEdgeHeuristic(hashesA);

//     std::cout << "Combined Edge Hashes for matrix A:" << std::endl;
//     for (size_t hash : combinedEdgeHashesA) {
//         std::cout << "Hash: " << hash << std::endl;
//     }

//     std::cout << std::endl;

//     auto combinedEdgeHashesB = combinedEdgeHeuristic(hashesB);

//     std::cout << "Combined Edge Hashes for matrix A:" << std::endl;
//     for (size_t hash : combinedEdgeHashesB) {
//         std::cout << "Hash: " << hash << std::endl;
//     }
    

//     return 0;
// }
