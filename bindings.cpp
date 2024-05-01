#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include "main.h"

namespace py = pybind11;

arma::Mat<long long> numpy_to_arma(py::array_t<long long, py::array::c_style | py::array::forcecast> array) {
    py::buffer_info buf = array.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be two");
    }

    arma::Mat<long long> mat(buf.shape[0], buf.shape[1], arma::fill::zeros);

    long long* ptr = static_cast<long long*>(buf.ptr);
    for (size_t i = 0; i < buf.shape[0]; i++) {
        for (size_t j = 0; j < buf.shape[1]; j++) {
            mat(i, j) = ptr[i * buf.strides[0]/sizeof(long long) + j];
        }
    }

    return mat;
}

// PYBIND11_MODULE(isohash, m) {
//     m.def("nodeHashCompare", [](py::array_t<long long> arrayA, py::array_t<long long> arrayB, int n) {
//         arma::Mat<long long> matA = numpy_to_arma(arrayA);
//         arma::Mat<long long> matB = numpy_to_arma(arrayB);
//         return combinedNodeHeuristic(matA, matB, n);
//     }, "A function that uses path counting to identify when two graphs are not isomorphic.");
// }
PYBIND11_MODULE(isohash, m) {
    m.def("nodeHashCompare", [](py::array_t<long long> arrayA, py::array_t<long long> arrayB, int n) {
        arma::Mat<long long> matA = numpy_to_arma(arrayA);
        arma::Mat<long long> matB = numpy_to_arma(arrayB);
        return nodeHashCompare(matA, matB, n);
    }, "A function that uses path counting based node isomorphism to identify when two graphs are not isomorphic.");

    m.def("edgeHashCompare", [](py::array_t<long long> arrayA, py::array_t<long long> arrayB, int n) {
        arma::Mat<long long> matA = numpy_to_arma(arrayA);
        arma::Mat<long long> matB = numpy_to_arma(arrayB);
        return edgeHashCompare(matA, matB, n);
    }, "A function that uses path counting based edge isomorphism to identify when two graphs are not isomorphic.");
}