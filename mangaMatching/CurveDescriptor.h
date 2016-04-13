#ifndef _CURVE_DESCRIPTOR_
#define _CURVE_DESCRIPTOR_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

class CurveDescriptor{
public:
	CurveDescriptor(std::vector<cv::Point2d> c, int sample_num, double s = 3.0, bool is_open = true);
	std::vector<cv::Point2d> get_sample_curve();
	std::vector<cv::Point2d> get_border_curve();
	std::vector<cv::Point2d> get_smooth_curve();
	std::vector<double> get_curvature();
	std::vector<double> get_curvature_integration();

private:
	double sigma;
	int M;
	std::vector<cv::Point2d> init_curve, curve, border_curve;
	std::vector<double> g, dg, ddg;
	std::vector<cv::Point2d> gp, dgp, ddgp;
	std::vector<double> curvature, curvature_integration;
	
	void bordering_curve();
	void sampling_curve(int sample_num);
	void gaussian_derivs();
	void p_gaussian(int index, bool is_open = true);
	void curve_gaussian(bool is_open = true);
	void curve_curvature();
	void integral_curvature();
};

#endif