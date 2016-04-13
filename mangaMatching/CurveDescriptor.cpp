#include "CurveDescriptor.h"

CurveDescriptor::CurveDescriptor(std::vector<cv::Point2d> c, int sample_num, double s, bool is_open){
	init_curve = curve = c;
	sigma = s;
	M = (int)round(10.0 * sigma + 1.0) / 2 * 2 - 1;
	assert(M % 2 == 1);

	sampling_curve(sample_num);
	if (is_open) bordering_curve();
	gaussian_derivs();
	curve_gaussian(is_open);
	curve_curvature();
	integral_curvature();
}

std::vector<cv::Point2d> CurveDescriptor::get_sample_curve(){
	return curve;
}

std::vector<cv::Point2d> CurveDescriptor::get_border_curve(){
	return border_curve;
}

std::vector<cv::Point2d> CurveDescriptor::get_smooth_curve(){
	return gp;
}

std::vector<double> CurveDescriptor::get_curvature(){
	return curvature;
}

std::vector<double> CurveDescriptor::get_curvature_integration(){
	return curvature_integration;
}

void CurveDescriptor::bordering_curve(){
	int L = M >> 1;
	border_curve.resize(curve.size() + L * 2);
	cv::Point2d v = curve[0] - curve[2];
	for (int i = 1; i <= L; ++i) border_curve[L - i] = curve[0] + v * i;
	for (unsigned int i = 0; i < curve.size(); ++i) border_curve[i + L] = curve[i];
	v = curve[curve.size() - 1] - curve[curve.size() - 3];
	for (int i = 1; i <= L; ++i) border_curve[L + curve.size() - 1 + i] = curve[curve.size() - 1] + v * i;
}

void CurveDescriptor::sampling_curve(int sample_num){
	double sample_length = 0;
	for (unsigned int i = 1; i < curve.size(); ++i)
		sample_length += cv::norm(curve[i] - curve[i - 1]);
	sample_length /= sample_num;

	std::vector<cv::Point2d> sample_curve;
	sample_curve.push_back(curve[0]);

	double curr_length = 0;
	for (unsigned int i = 1; i < curve.size(); ++i){
		double length = cv::norm(curve[i] - curve[i - 1]);
		curr_length += length;
		while (curr_length > sample_length){
			curr_length -= sample_length;
			// i - 1 |------X----curr_length----| i
			sample_curve.push_back(cv::Point2d((curve[i].x * (length - curr_length) + curve[i - 1].x * curr_length) / length,
				(curve[i].y * (length - curr_length) + curve[i - 1].y * curr_length) / length));
		}
	}
	if (curr_length < sample_length * 0.5) sample_curve.erase(sample_curve.end() - 1);
	sample_curve.push_back(curve[curve.size() - 1]);
	curve = sample_curve;
	return;
}

// 計算-L~L之間的高斯分布
// http://www.cedar.buffalo.edu/~srihari/CSE555/Normal2.pdf
void CurveDescriptor::gaussian_derivs(){
	g.resize(M), dg.resize(M), ddg.resize(M);
	int L = M >> 1;
	double sigma_squre = sigma * sigma;
	double sigma_quad = sigma_squre * sigma_squre;

	cv::Mat gaussian_kernal = cv::getGaussianKernel(M, sigma, CV_64F);
	for (int i = -L; i <= L; i ++){
		g[i + L] = gaussian_kernal.at<double>(i + L);
		dg[i + L] = -i * 1.0 / sigma_squre * gaussian_kernal.at<double>(i + L);
		ddg[i + L] = (-sigma_squre + i * 1.0 * i) / sigma_quad * gaussian_kernal.at<double>(i + L);
	}	
	return;
}

void CurveDescriptor::p_gaussian(int index, bool is_open){
	int L = M >> 1;

	gp[index] = dgp[index] = ddgp[index] = cv::Point2d(0, 0);
	for (int i = -L; i <= L; ++i){
		cv::Point2d p;
		if (is_open) p = border_curve[index - i + L];
		else p = curve[(index - i + curve.size()) % curve.size()];

		gp[index] += p * g[i + L];
		dgp[index] += p * dg[i + L];
		ddgp[index] += p * ddg[i + L];
	}
	return;
}

void CurveDescriptor::curve_gaussian(bool is_open){
	gp.resize(curve.size()), dgp.resize(curve.size()), ddgp.resize(curve.size());
	for (int i = 0; i < curve.size(); ++i)
		p_gaussian(i, is_open);
	return;
}

void CurveDescriptor::curve_curvature(){
	curvature.resize(curve.size());
	for (int i = 0; i < curve.size(); i++) // Mokhtarian 02' eqn (4)
		curvature[i] = (dgp[i].x * ddgp[i].y - ddgp[i].x * dgp[i].y) / pow(dgp[i].x * dgp[i].x + dgp[i].y * dgp[i].y, 1.5);
}

void CurveDescriptor::integral_curvature(){
	curvature_integration.push_back(0);
	for (unsigned int i = 1; i < curve.size(); ++i)
		curvature_integration.push_back((curvature[i - 1] + curvature[i]) / 2 * cv::norm(curve[i] - curve[i - 1]));
	return;
}