#include "CurveDescriptor.h"

CurveDescriptor::CurveDescriptor(){
	curve.clear();
	border_curve.clear();
	smooth_curve.clear();
	curvature.clear();
	scale_curvature.clear();
}

// not need sampling
CurveDescriptor::CurveDescriptor(std::vector<cv::Point2d> c, double s, bool is_open){
	init_curve = curve = c;
	sigma = s;
	M = (int)round(10.0 * sigma + 1.0) / 2 * 2 - 1;
	assert(M % 2 == 1);
	gaussian_derivs();

	if (curve.size() <= 3){
		too_short_error = true;
		return;
	}
	if (is_open){
		border_curve = bordering_curve(curve);
		curve_gaussian(border_curve, c_gp, c_dgp, c_ddgp, is_open);
	}
	else curve_gaussian(curve, c_gp, c_dgp, c_ddgp, is_open);
	smooth_curve = c_gp;
	curvature = curve_curvature(curve, c_dgp, c_ddgp); // abs curvature
}

CurveDescriptor::CurveDescriptor(std::vector<cv::Point2d> c, double sample_length, double s, bool is_open){
	init_curve = c;
	sigma = s;
	M = (int)round(10.0 * sigma + 1.0) / 2 * 2 - 1;
	assert(M % 2 == 1);
	gaussian_derivs();

	if (!is_open && c.back() != c[0]) c.push_back(c[0]);
	curve = sampling_curve(c, sample_length);
	if (!is_open) curve.pop_back();
	if (curve.size() <= 3){
		too_short_error = true;
		return;
	}

	if (is_open){
		border_curve = bordering_curve(curve);
		curve_gaussian(border_curve, c_gp, c_dgp, c_ddgp, is_open);
	}
	else curve_gaussian(curve, c_gp, c_dgp, c_ddgp, is_open);

	smooth_curve = c_gp;
	curvature = curve_curvature(curve, c_dgp, c_ddgp); // abs curvature
}

CurveDescriptor::CurveDescriptor(std::vector<cv::Point2d> c, unsigned int sample_num, double s, bool is_open){
	init_curve = curve = c;
	sigma = s;
	M = (int)round(10.0 * sigma + 1.0) / 2 * 2 - 1;
	assert(M % 2 == 1);
	gaussian_derivs();

	curve = sampling_curve(c, sample_num);
	if (is_open){
		border_curve = bordering_curve(curve);
		curve_gaussian(border_curve, c_gp, c_dgp, c_ddgp, is_open);
	}
	else curve_gaussian(curve, c_gp, c_dgp, c_ddgp, is_open);
	smooth_curve = c_gp;
	curvature = curve_curvature(curve, c_dgp, c_ddgp); // abs curvature
}

std::vector<double> CurveDescriptor::scaling_curvature(double s){
	scale_curvature.clear();
	for (double i = 0; i <= curvature.size() - 1; i += 1 / s){
		if (i == curvature.size() - 1) scale_curvature.push_back(curvature[i]);
		else scale_curvature.push_back(curvature[(int)i] * (floor(i) + 1 - i) + curvature[(int)i + 1] * (i - floor(i)));
	}
	return scale_curvature;
}

std::vector<cv::Point2d> CurveDescriptor::get_segment_curve(unsigned int begin, unsigned int end){
	std::vector<cv::Point2d> segment;
	double begin_x = integration_sample[begin + 1].x;
	double end_x = integration_sample[end + 1].x;
	int p = floor(begin_x) / 2 - 1;
	int q = ceil(end_x) / 2 - 1;
	for (int i = p; i <= q; ++i){
		segment.push_back(curve[i]);
	}
	return segment;
}

bool CurveDescriptor::is_error(){
	if (too_short_error) return true;
	return false;
}

std::vector<cv::Point2d> CurveDescriptor::bordering_curve(std::vector<cv::Point2d> c){
	int L = M >> 1;
	std::vector<cv::Point2d> b;
	b.resize(c.size() + L * 2);
	cv::Point2d v = c[0] - c[2];
	for (int i = 1; i <= L; ++i) b[L - i] = c[0] + v * i;
	for (unsigned int i = 0; i < c.size(); ++i) b[i + L] = c[i];
	v = c[c.size() - 1] - c[c.size() - 3];
	for (int i = 1; i <= L; ++i) b[L + c.size() - 1 + i] = c[c.size() - 1] + v * i;
	return b;
}

std::vector<cv::Point2d> CurveDescriptor::sampling_curve(std::vector<cv::Point2d> c, double sample_length){
	std::vector<cv::Point2d> sample_curve;
	sample_curve.push_back(c[0]);
	double curr_length = 0;
	for (unsigned int i = 1; i < c.size(); ++i){
		double length = cv::norm(c[i] - c[i - 1]);
		curr_length += length;
		while (curr_length > sample_length){
			curr_length -= sample_length;
			// i - 1 |------X----curr_length----| i
			sample_curve.push_back(cv::Point2d((c[i].x * (length - curr_length) + c[i - 1].x * curr_length) / length,
				(c[i].y * (length - curr_length) + c[i - 1].y * curr_length) / length));
		}
	}
	if (curr_length < sample_length * 0.5) sample_curve.erase(sample_curve.end() - 1);
	sample_curve.push_back(c[c.size() - 1]);
	return sample_curve;
}

std::vector<cv::Point2d> CurveDescriptor::sampling_curve(std::vector<cv::Point2d> c, unsigned int sample_num){
	double sample_length = 0;
	for (unsigned int i = 1; i < curve.size(); ++i)
		sample_length += cv::norm(curve[i] - curve[i - 1]);
	sample_length /= sample_num;

	return sampling_curve(c, sample_length);
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

void CurveDescriptor::p_gaussian(std::vector<cv::Point2d> c, int index, std::vector<cv::Point2d> &gp, std::vector<cv::Point2d> &dgp, std::vector<cv::Point2d> &ddgp, bool is_open){
	int L = M >> 1;

	gp[index] = dgp[index] = ddgp[index] = cv::Point2d(0, 0);
	for (int i = -L; i <= L; ++i){
		cv::Point2d p;
		if (is_open) p = c[index - i + L];
		else p = c[(index - i + c.size()) % c.size()];

		gp[index] += p * g[i + L];
		dgp[index] += p * dg[i + L];
		ddgp[index] += p * ddg[i + L];
	}
	return;
}

void CurveDescriptor::curve_gaussian(std::vector<cv::Point2d> c, std::vector<cv::Point2d> &gp, std::vector<cv::Point2d> &dgp, std::vector<cv::Point2d> &ddgp, bool is_open){
	int L = is_open ? M >> 1 : 0;
	gp.resize(c.size() - 2 * L), dgp.resize(c.size() - 2 * L), ddgp.resize(c.size() - 2 * L);
	for (int i = 0; i < c.size() - 2 * L; ++i)
		p_gaussian(c, i, gp, dgp, ddgp, is_open);
	return;
}

std::vector<double> CurveDescriptor::curve_curvature(std::vector<cv::Point2d> c, std::vector<cv::Point2d> dgp, std::vector<cv::Point2d> ddgp){
	std::vector<double> crvt;
	crvt.resize(c.size());
	for (int i = 0; i < c.size(); i++) // Mokhtarian 02' eqn (4)
		crvt[i] = abs(dgp[i].x * ddgp[i].y - ddgp[i].x * dgp[i].y) / pow(dgp[i].x * dgp[i].x + dgp[i].y * dgp[i].y, 1.5);
	return crvt;
}

void CurveDescriptor::caculate_integration(){
	integration.resize(curvature.size() * 2 + 1);
	for (unsigned int i = 0; i < integration.size(); ++i){
		if (i == 0)
			integration[i] = cv::Point2d(i, 0);
		else if (i == 1)
			integration[i] = cv::Point2d(i, abs(curvature[i]) / 2);
		else if (!(i % 2))
			integration[i] = cv::Point2d(i, integration[i - 1].y + abs(curvature[i / 2 - 1]));
		else if (i % 2)
			integration[i] = cv::Point2d(i, integration[i - 1].y + (abs(curvature[i / 2 - 1]) + abs(curvature[i / 2]) / 2));
	}
	return;
}

void CurveDescriptor::sampling_integration(){
	double sample_y = 1.0;
	double curr_y = 0;
	unsigned int i = 1;
	integration_sample.push_back(cv::Point2d(0, 0));
	curr_y += sample_y;
	while (i < integration.size()){
		if (curr_y >= integration[i - 1].y && curr_y < integration[i].y){
			integration_sample.push_back(cv::Point2d(
				(integration[i].x * (curr_y - integration[i - 1].y) + integration[i - 1].x * (integration[i].y - curr_y)) / (integration[i].y - integration[i - 1].y),
				curr_y));
			curr_y += sample_y;
		}
		else ++i;
	}
	return;
}

std::vector<double> CurveDescriptor::intg_curvature(std::vector<cv::Point2d> c, std::vector<cv::Point2d> dgp, std::vector<cv::Point2d> ddgp){
	int L = M >> 1;
	std::vector<double> crvt;
	crvt.resize(c.size() - L * 2);
	for (int i = 0; i < crvt.size(); i++) // Mokhtarian 02' eqn (4)
		crvt[i] = (dgp[i].x * ddgp[i].y - ddgp[i].x * dgp[i].y) / pow(dgp[i].x * dgp[i].x + dgp[i].y * dgp[i].y, 1.5);
	return crvt;
}