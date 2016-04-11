#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "VectorCurve.h"


template<typename T>
// type:Mat type ex: uchar(0), i: row, j: col, c: channel
T &get_Mat_val(cv::Mat &m, T type, int i, int j, int c){
	return ((T *)m.data)[(i * m.cols + j) * m.channels() + c];
}

cv::RNG rng = cv::RNG(7236);
int maint(){
	char *filename = "data/0000.png";
	cv::Mat img_read = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::Mat img_gray, img_gray_32f;
	cv::Mat img_draw = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255)), img_resize;
	cv::cvtColor(img_read, img_gray, CV_BGR2GRAY);
	img_gray.convertTo(img_gray_32f, CV_32FC1, 1.0 / 255);

	std::vector<std::vector<cv::Point>> edges;
	VectorCurve dEdge(img_gray_32f);
	dEdge.CalSecDer();
	edges = dEdge.Link();

	printf("find %d curve done.\n", edges.size());
	for (size_t i = 0; i < edges.size(); ++i){
		cv::Scalar color = cv::Scalar(rng.uniform(0, 221), rng.uniform(0, 221), rng.uniform(0, 221));
		for (size_t j = 1; j < edges[i].size(); ++j){
			cv::line(img_draw, edges[i][j - 1], edges[i][j], color);
		}
	}

	cv::resize(img_draw, img_resize, cv::Size(img_draw.cols * 3, img_draw.rows * 3));
	cv::imshow("draw", img_resize);
	cv::waitKey(0);

	return 0;
}