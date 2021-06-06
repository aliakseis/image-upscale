// image-upscale.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <lbfgs.h>

#define USE_FFTW 0

#if USE_FFTW
#include <fftw3.h>
#endif

#include <algorithm>
#include <iostream>
#include <random>
#include <map>
#include <vector>
#include <exception>



struct LbfgsContext {
    cv::Size originalSize;
    const cv::Mat& b;
};

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
    auto context = static_cast<LbfgsContext*>(instance);

    const cv::Mat x2(context->originalSize.height * 2,
        context->originalSize.width * 2,
        CV_64FC1,
        const_cast<void*>(static_cast<const void*>(x)));

    cv::Mat Ax2;
    cv::idct(x2, Ax2);

    double fx = 0;

    for (int y = 0; y < context->originalSize.height; ++y)
        for (int x = 0; x < context->originalSize.width; ++x)
        {
            const auto v = (Ax2.at<double>(y * 2, x * 2)
                + Ax2.at<double>(y * 2, x * 2 + 1)
                + Ax2.at<double>(y * 2 + 1, x * 2)
                + Ax2.at<double>(y * 2 + 1, x * 2 + 1)) / 4;

            const auto Ax = v - context->b.at<uchar>(y, x);

            Ax2.at<double>(y * 2, x * 2)
                = Ax2.at<double>(y * 2, x * 2 + 1)
                = Ax2.at<double>(y * 2 + 1, x * 2)
                = Ax2.at<double>(y * 2 + 1, x * 2 + 1)
                = Ax;

            fx += Ax * Ax * 4;
        }

    cv::Mat AtAxb2(context->originalSize.height * 2,
        context->originalSize.width * 2,
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
        cv::Mat src = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        imshow("Original", src);


        LbfgsContext context{ {src.cols, src.rows}, src };

        const double param_c = 5;

        const int numImgPixels = src.rows * src.cols * 4;

        // Initialize solution vector
        lbfgsfloatval_t fx;
        lbfgsfloatval_t *x = lbfgs_malloc(numImgPixels);
        if (x == nullptr) {
            //
        }
        for (int i = 0; i < numImgPixels; i++) {
            x[i] = 1;
        }

        // Initialize the parameters for the optimization.
        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.orthantwise_c = param_c; // this tells lbfgs to do OWL-QN
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
        int lbfgs_ret = lbfgs(numImgPixels, x, &fx, evaluate, progress, &context, &param);

        cv::Mat Xat2(context.originalSize.height * 2, context.originalSize.width * 2, CV_64FC1, x);
        cv::Mat Xa;
        idct(Xat2, Xa);

        lbfgs_free(x);

        cv::Mat dst;
        Xa.convertTo(dst, CV_8U);

        imshow("Restored", dst);

        cv::waitKey();

    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';
    }
}