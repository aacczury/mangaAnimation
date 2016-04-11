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

#include "VectorCurve.h"

using namespace System::Drawing;

namespace std{
	template <> struct hash<cv::Point>{
		size_t operator()(cv::Point const &p) const{
			return 53 + std::hash<int>()(p.x) * 53 + std::hash<int>()(p.y);
		}
	};
}

class mangaShow{
public:
	mangaShow(){};
	mangaShow(char *);
	void vector_curves();
	void remove_dump_by_ROI(unsigned short thickness = 2);
	void topol_curves();
	void link_adjacent();

	void rng_curves_color();
	void set_curves_drawable(int index = -1, bool is_draw = true);
	void draw_curves();
	void draw_topol();
	
	Bitmap ^mat2Bitmap(cv::Mat);
	Bitmap ^get_canvas_Bitmap();
	std::vector<bool> get_curves_drawable();

	void test();

private:
	int scale = 3;
	cv::RNG rng = cv::RNG(1235);
	cv::Mat img_read, img_show, img_show_scale;
	std::vector<std::vector<cv::Point>> curves;
	std::vector<cv::Scalar> curves_color;
	std::vector<bool> curves_drawable;
	std::unordered_set<cv::Point> curves_pnts;
	std::unordered_map<cv::Point, unordered_set<cv::Point>> topol;

	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar yellow = cv::Scalar(0, 200, 200);
	cv::Scalar green = cv::Scalar(0, 255, 0);
	cv::Scalar cyan = cv::Scalar(200, 200, 0);
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar gray = cv::Scalar(125, 125, 125);

	template<typename T>
	// type:Mat type ex: uchar(0), i: row, j: col, c: channel
	T &ref_Mat_val(cv::Mat &m, T type, int i, int j, int c = 0);

	template<typename T>
	// type:Mat type ex: uchar(0), p: Point, c: channel
	T &ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c = 0);

	void compute_curvature(std::vector<cv::Point>);
};

#endif