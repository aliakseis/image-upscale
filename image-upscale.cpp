// image-upscale.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <lbfgs.h>


#include <algorithm>
#include <iostream>
#include <vector>
#include <exception>


static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
)
{
    return 0;
}

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
)
{
    auto src = static_cast<cv::Mat*>(instance);

    const cv::Mat x2(src->rows * 2,
        src->cols * 2,
        CV_64FC1,
        const_cast<void*>(static_cast<const void*>(x)));

    cv::Mat Ax2;
    cv::idct(x2, Ax2);

    double fx = 0;

    for (int y = 0; y < src->rows; ++y)
        for (int x = 0; x < src->cols; ++x)
        {
            const auto v = (Ax2.at<double>(y * 2, x * 2)
                + Ax2.at<double>(y * 2, x * 2 + 1)
                + Ax2.at<double>(y * 2 + 1, x * 2)
                + Ax2.at<double>(y * 2 + 1, x * 2 + 1)) / 4;

            const auto Ax = v - src->at<uchar>(y, x);

            Ax2.at<double>(y * 2, x * 2)
                = Ax2.at<double>(y * 2, x * 2 + 1)
                = Ax2.at<double>(y * 2 + 1, x * 2)
                = Ax2.at<double>(y * 2 + 1, x * 2 + 1)
                = Ax;

            fx += Ax * Ax * 4;
        }

    cv::Mat AtAxb2(src->rows * 2,
        src->cols * 2,
        CV_64FC1,
        g);
    cv::dct(Ax2, AtAxb2);
    AtAxb2 *= 2;

    return fx;
};

int main(int argc, char** argv)
{
    try {
        cv::String filename;
        if (argc >= 2)
            filename = argv[1];
        else {
            try {
                filename = cv::samples::findFile("lena.jpg");
            }
            catch (const std::exception& ex) {
                std::string s(ex.what());
                s = s.substr(s.find(") ") + 2);
                s = s.substr(0, s.find("modules"));
                filename = s + "samples/data/lena.jpg";
            }
        }
        cv::Mat img = cv::imread(filename);

        imshow("Original", img);

        std::vector<cv::Mat> bgr;
        split(img, bgr);

        for (auto& src : bgr)
        {
            const double param_c = 5;

            const int numImgPixels = src.rows * src.cols * 4;

            // Initialize solution vector
            lbfgsfloatval_t fx;
            lbfgsfloatval_t *x = lbfgs_malloc(numImgPixels);
            if (x == nullptr) {
                return EXIT_FAILURE;
            }
            for (int i = 0; i < numImgPixels; i++) {
                x[i] = 1;
            }

            // Initialize the parameters for the optimization.
            lbfgs_parameter_t param;
            lbfgs_parameter_init(&param);
            param.orthantwise_c = param_c; // this tells lbfgs to do OWL-QN
            param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
            int lbfgs_ret = lbfgs(numImgPixels, x, &fx, evaluate, progress, &src, &param);

            cv::Mat Xat2(src.rows * 2, src.cols * 2, CV_64FC1, x);
            cv::Mat Xa;
            idct(Xat2, Xa);

            lbfgs_free(x);

            Xa.convertTo(src, CV_8U);
        }

        cv::Mat dst;
        merge(bgr, dst);
        imshow("Result", dst);

        cv::waitKey();

        if (argc >= 3)
            cv::imwrite(argv[2], dst);
    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';
    }
}
