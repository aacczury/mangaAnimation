#ifndef _MANGASHOW_H_
#define _MANGASHOW_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <Windows.h>

//#include "VectorCurve.h"
#include "CurveDescriptor.h"

using namespace System::Drawing;

#ifndef _POINT2D_HASH_
#define _POINT2D_HASH_
namespace std{ // cv::Point unordered_set and unordered_map need a hash function
	template <> struct hash<cv::Point2d>{
		size_t operator()(cv::Point2d const &p) const{
			return 53 + std::hash<double>()(p.x) * 53 + std::hash<double>()(p.y);
		}
	};
}
#endif

class mangaShow{
public:
	mangaShow();
	void read_img(char *);
	void read_graph(char *);
	void build_curves();
	void caculate_curve();

	void draw_graph();
	void rng_curves_color();
	void set_curves_drawable(int index = -1, bool is_draw = true);
	void draw_curves();

	bool is_read_img();
	bool is_read_graph();
	Bitmap ^mat2Bitmap(cv::Mat);
	Bitmap ^get_canvas_Bitmap();
	std::vector<bool> get_curves_drawable();

	void test();

private:
	LARGE_INTEGER start_t, end_t, freq;

	int scale = 3;
	cv::RNG rng = cv::RNG(1235);
	cv::Mat img_read, img_show, canvas;
	std::unordered_set<cv::Point2d> graph_pnts;
	std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> graph;
	std::unordered_set<cv::Point2d> end_pnts;
	std::unordered_set<cv::Point2d> junction_pnts;
	std::vector<std::vector<cv::Point2d>> curves;
	std::vector<cv::Scalar> curves_color;
	std::vector<bool> curves_drawable;

	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar yellow = cv::Scalar(0, 200, 200);
	cv::Scalar green = cv::Scalar(0, 255, 0);
	cv::Scalar cyan = cv::Scalar(200, 200, 0);
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar purple = cv::Scalar(200, 0, 200);
	cv::Scalar gray = cv::Scalar(125, 125, 125);

	std::vector<cv::Point2d> link_curve(cv::Point2d p, cv::Point2d q, std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> &pnts_used_pnts);
	unsigned int normalize_cross_correlation(std::vector<double> a, std::vector<double> b);
	void draw_plot_graph(std::vector<cv::Point2d> data);

	template<typename T>
	// type:Mat type ex: uchar(0), i: row, j: col, c: channel
	T &ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c = 0){
		return ((T *)m.data)[(p.y * m.cols + p.x) * m.channels() + c];
	}
};

#endif