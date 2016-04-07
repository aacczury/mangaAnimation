#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "CmCurveEx.h"


template<typename T>
// type:Mat type ex: uchar(0), i: row, j: col, c: channel
T &get_Mat_val(cv::Mat &m, T type, int i, int j, int c){
	return ((T *)m.data)[(i * m.cols + j) * m.channels() + c];
}

cv::RNG rng = cv::RNG(7236);
int main(){
	char *filename = "data/0000.png";
	cv::Mat img_read = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::Mat img_gray, img_gray_32f;
	// img_show = img_read.clone();
	cv::cvtColor(img_read, img_gray, CV_BGR2GRAY);
	img_gray.convertTo(img_gray_32f, CV_32FC1, 1.0 / 255);

	std::vector<CmCurveEx::CEdge> edges;
	CmCurveEx dEdge(img_gray_32f);
	dEdge.CalSecDer();
	dEdge.Link();
	edges = dEdge.GetEdges();
	
	printf("find %d curve done.\n", edges.size());
	cv::Mat img_draw = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (size_t i = 0; i < edges.size(); ++i){
		cv::Scalar color = cv::Scalar(rng.uniform(0, 221), rng.uniform(0, 221), rng.uniform(0, 221));
		for (size_t j = 1; j < edges[i].pnts.size(); ++j){
			cv::line(img_draw, edges[i].pnts[j - 1], edges[i].pnts[j], color);
		}
	}
	cv::imshow("draw", img_draw);
	cv::waitKey(0);

	return 0;
}