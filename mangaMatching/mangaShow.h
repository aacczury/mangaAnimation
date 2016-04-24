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

#include "GraphFile.h"
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

#define MANGA_FACE 0
#define SAMPLE_FACE 1

class mangaShow{
public:
	mangaShow();
	void read_img(char *);
	void read_graph(char *, int g_s = MANGA_FACE);

	void find_seed();

	void draw_graph();
	void rng_curves_color();
	void set_curves_drawable(int index = -1, bool is_draw = true);
	void draw_curves(bool is_colorful = true);

	bool is_read_img();
	bool is_read_mangaFace();
	bool is_read_sampleFace();

	Bitmap ^mat2Bitmap(cv::Mat);
	Bitmap ^get_canvas_Bitmap();
	Bitmap ^get_sample_canvas_Bitmap();
	std::vector<bool> get_curves_drawable();

	void test();

private:
	LARGE_INTEGER start_t, end_t, freq;

	int scale = 3;
	cv::RNG rng = cv::RNG(1235);
	cv::Mat img_read;
	cv::Mat img_show, canvas;
	cv::Mat sample_show, sample_canvas;
	std::vector<cv::Scalar> curves_color;
	std::vector<bool> curves_drawable;

	GraphFile mangaFace, sampleFace;
	std::vector<CurveDescriptor> mangaFace_CD, sampleFace_CD;
	
	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar yellow = cv::Scalar(0, 200, 200);
	cv::Scalar green = cv::Scalar(0, 255, 0);
	cv::Scalar cyan = cv::Scalar(200, 200, 0);
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar purple = cv::Scalar(200, 0, 200);
	cv::Scalar gray = cv::Scalar(125, 125, 125);

	int normalize_cross_correlation(std::vector<double> a, std::vector<double> b);
	void compare_curve(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b);
	void compare_curves(std::vector<cv::Point2d> sample_curve);

	template<typename T> // type:Mat type ex: uchar(0), i: row, j: col, c: channel
	T &ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c = 0);
	double curve_length(std::vector<cv::Point2d> curve);
	double perpendicular_distance(cv::Point2d p, cv::Point2d p1, cv::Point2d p2);
	std::vector<unsigned int> douglas_peucker(std::vector<cv::Point2d> &line, int max_depth, int p = 0, int q = -1, int depth = 0);
	unsigned int max_curvature_index(std::vector<double> curvature);
	double abc_degree(cv::Point2d a, cv::Point2d b, cv::Point2d c);
	cv::Point2d curve_pnt_tangent(std::vector<cv::Point2d> curve, unsigned int index);

	void draw_sample_face(unsigned int sample);
	void draw_plot_graph(std::vector<cv::Point2d> data, char *win_name);
	void draw_plot_graph(std::vector<cv::Point2d> data_a, std::vector<cv::Point2d> data_b, double offset, char *win_name);
};

#endif