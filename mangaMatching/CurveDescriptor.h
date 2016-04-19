#ifndef _CURVE_DESCRIPTOR_
#define _CURVE_DESCRIPTOR_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

class CurveDescriptor{
public:
	CurveDescriptor(std::vector<cv::Point2d> c, double sample_length, double s = 3.0, bool is_open = true);
	CurveDescriptor(std::vector<cv::Point2d> c, unsigned int sample_num, double s = 3.0, bool is_open = true);

	std::vector<cv::Point2d> get_sample_curve();
	std::vector<cv::Point2d> get_border_curve();
	std::vector<cv::Point2d> get_smooth_curve();
	std::vector<double> get_curvature();

	std::vector<cv::Point2d> get_integration();
	std::vector<cv::Point2d> get_integration_sample();
	std::vector<cv::Point2d> get_smooth_integration();
	std::vector<double> get_integration_curvature();

	std::vector<cv::Point2d> get_segment_curve(unsigned int begin, unsigned int end);

private:
	double sigma;
	int M;
	std::vector<double> g, dg, ddg;
	std::vector<cv::Point2d> init_curve;

	std::vector<cv::Point2d> curve, border_curve;
	std::vector<cv::Point2d> c_gp, c_dgp, c_ddgp;
	std::vector<double> curvature;

	std::vector<cv::Point2d> integration, integration_border, integration_sample;
	std::vector<cv::Point2d> i_gp, i_dgp, i_ddgp;
	std::vector<double> integration_curvature;
	
	void p_gaussian(std::vector<cv::Point2d> c, int index, std::vector<cv::Point2d> &gp, std::vector<cv::Point2d> &dgp, std::vector<cv::Point2d> &ddgp, bool is_open = true);
	void curve_gaussian(std::vector<cv::Point2d> c, std::vector<cv::Point2d> &gp, std::vector<cv::Point2d> &dgp, std::vector<cv::Point2d> &ddgp, bool is_open = true);
	std::vector<double> curve_curvature(std::vector<cv::Point2d> c, std::vector<cv::Point2d> dgp, std::vector<cv::Point2d> ddgp);
	
	void gaussian_derivs();
	std::vector<cv::Point2d> sampling_curve(std::vector<cv::Point2d> c, double sample_length);
	std::vector<cv::Point2d> sampling_curve(std::vector<cv::Point2d> c, unsigned int sample_num);
	std::vector<cv::Point2d> bordering_curve(std::vector<cv::Point2d> c);
	std::vector<double> intg_curvature(std::vector<cv::Point2d> c, std::vector<cv::Point2d> dgp, std::vector<cv::Point2d> ddgp);

	void caculate_integration();
	void sampling_integration();
};

#endif