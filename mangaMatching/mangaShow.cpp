#include "mangaShow.h"

mangaShow::mangaShow(){
	img_read = img_show = canvas = cv::Mat();
	curves.clear();
	curves_color.clear();
	curves_drawable.clear();
	graph_pnts.clear();
	graph.clear();
}

void mangaShow::read_img(char *filename){
	img_read = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	// need exception handling
	img_show = img_read.clone();
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));

	QueryPerformanceFrequency(&freq);

	printf("=> Reading image done.\n");
}

void mangaShow::read_graph(char *filename){
	FILE *graph_file = fopen(filename, "r");
	std::vector<cv::Point2d> graph_pnts_vector;
	graph_pnts_vector.clear(), graph_pnts.clear(), graph.clear();

	unsigned int pnts_size;
	fscanf(graph_file, "%u\n", &pnts_size);
	for (unsigned int i = 0; i < pnts_size; ++i){
		cv::Point2d p;
		fscanf(graph_file, "V %lf %lf\n", &p.x, &p.y);
		graph_pnts_vector.push_back(p);
		graph_pnts.insert(p);
	}

	unsigned int edges_size;
	fscanf(graph_file, "%u\n", &edges_size);
	for (unsigned int i = 0; i < edges_size; ++i){
		unsigned int p, q;
		fscanf(graph_file, "S %u %u\n", &p, &q);
		graph[graph_pnts_vector[p]].insert(graph_pnts_vector[q]);
		graph[graph_pnts_vector[q]].insert(graph_pnts_vector[p]);
	}

	printf("=> Reading graph done.\n");
	return;
}

void mangaShow::read_match_curve(char *filename){
	FILE *curve_file = fopen(filename, "r");
	match_curve.clear();

	unsigned int pnts_size;
	fscanf(curve_file, "%u\n", &pnts_size);
	for (unsigned int i = 0; i < pnts_size; ++i){
		cv::Point2d p;
		fscanf(curve_file, "V %lf %lf\n", &p.x, &p.y);
		match_curve.push_back(p);
	}
	
	printf("=> Reading match curve done.\n");
	return;
}

bool curve_length_cmp(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b){
	double a_length = 0, b_length = 0;
	for (unsigned int i = 1; i < a.size(); ++i) a_length += cv::norm(a[i] - a[i - 1]);
	for (unsigned int i = 1; i < b.size(); ++i) b_length += cv::norm(b[i] - b[i - 1]);
	return a_length > b_length;
}
void mangaShow::build_curves(){
	std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> pnts_used_pnts;
	for (const cv::Point2d &p : graph_pnts){
		if (graph[p].size() == 1){
			end_pnts.insert(p);
			pnts_used_pnts[p] = graph[p];
		}
		if (graph[p].size() > 2){
			junction_pnts.insert(p);
			pnts_used_pnts[p] = graph[p];
		}
	}
	
	for (const cv::Point2d p : end_pnts){
		cv::Point2d q = *(graph[p].begin());
		if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
		curves.push_back(link_curve(p, q, pnts_used_pnts));
	}

	for (const cv::Point2d p : junction_pnts){
		for (const cv::Point2d q : graph[p]){
			if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
			curves.push_back(link_curve(p, q, pnts_used_pnts));
		}
	}

	std::sort(curves.begin(), curves.end(), curve_length_cmp);
	printf("=> Building curves done\n");
	return;
}

void mangaShow::building_pnt_to_curve(){
	pnt_to_curve.clear();
	for (unsigned int i = 0; i < curves.size(); ++i){
		pnt_to_curve[curves[i][0]].push_back(i);
		pnt_to_curve[curves[i][curves[i].size() - 1]].push_back(i);
	}
	return;
}

void mangaShow::calculate_curve(){
	CurveDescriptor cd = CurveDescriptor(match_curve, 0.01, 3.0, true);
	std::vector<cv::Point2d> smooth_curve = cd.get_smooth_curve();
	std::vector<double> curvature = cd.get_curvature();
	std::vector<cv::Point2d> integration = cd.get_integration();
	std::vector<cv::Point2d> integration_sample = cd.get_integration_sample();
	std::vector<double> integration_curvature = cd.get_integration_curvature();

	std::vector<cv::Point2d> data;
	data.resize(curvature.size());
	for (unsigned int i = 0; i < curvature.size(); ++i) data[i] = cv::Point2d(i, abs(curvature[i]));
	draw_plot_graph(data, "match_curve_curvature");
	draw_plot_graph(integration, "match_curve_integration");
	draw_plot_graph(integration_sample, "match_curve_integration_sample");
	data.resize(integration_curvature.size());
	for (unsigned int i = 0; i < integration_curvature.size(); ++i){
		data[i] = cv::Point2d(i, abs(integration_curvature[i]));
		printf("%lf, %lf\n", data[i].x, data[i].y);
	}
	draw_plot_graph(data, "match_curve_integration_curvature");
}

std::vector<double> mangaShow::scale_curvature(std::vector<double> curvature, double s){
	std::vector<double> s_c;
	for(double i = 0; i <= curvature.size() - 1; i += 1 / s){
		if (i == curvature.size() - 1) s_c.push_back(curvature[i]);
		else s_c.push_back(curvature[(int)i] * (floor(i) + 1 - i) + curvature[(int)i + 1] * (i - floor(i)));
	}
	return s_c;
}

double mangaShow::curve_length(std::vector<cv::Point2d> curve){
	double total = 0;
	for (unsigned int i = 1; i < curve.size(); ++i)	total += cv::norm(curve[i] - curve[i - 1]);
	return total;
}

bool point2d_vector_size_cmp(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b){
	return a.size() > b.size();
}
void mangaShow::compare_curve(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b){
	double a_ratio = curve_length(a) / cv::norm(a[0] - a[a.size() - 1]);

	CurveDescriptor a_d = CurveDescriptor(a, 0.01, 3.0, true);
	if (a_d.is_error()) return;
	CurveDescriptor b_d = CurveDescriptor(b, 0.005, 3.0, true);
	if (b_d.is_error()) return;

	std::vector<double> a_crvt = a_d.get_curvature();
	//for (unsigned int i = 0; i < a_crvt.size(); ++i) printf(".. %lf\n", a_crvt[i]);
	std::vector<double> a_crvt_refine;
	for (unsigned int i = 0; i < a_crvt.size(); ++i) if (a_crvt[i] > 1.0) a_crvt_refine.push_back(a_crvt[i]);
	std::vector<double> b_crvt = b_d.get_curvature();

	//std::vector<cv::Point2d> data_a;
	//for (unsigned int i = 0; i < a_crvt.size(); ++i) data_a.push_back(cv::Point2d(i, a_crvt[i]));

	std::vector<cv::Point2d> b_sample = b_d.get_sample_curve();

	std::vector<unsigned int> repeat_longest_range;
	repeat_longest_range.resize(b_sample.size());
	for (double i = 1; i <= 4; ++i){
		if (a_crvt.size() / i <= 3) break;
		std::vector<double> b_crvt_scale = scale_curvature(b_crvt, i);
		if (b_crvt_scale.size() < a_crvt.size()) continue;

		int offset = normalize_cross_correlation(a_crvt, b_crvt_scale);
		//char s[100];
		//sprintf(s, "cc %i", i);
		//std::vector<cv::Point2d> data_b;
		//for (unsigned int i = 0; i < b_crvt_scale.size(); ++i) data_b.push_back(cv::Point2d(i, b_crvt_scale[i]));
		//draw_plot_graph(data_a, data_b, offset, s);

		if (offset < 0) continue;
		for (unsigned int j = 0; j < a_crvt.size() / i; ++j)
			repeat_longest_range[offset / i + j] ++;
	}

	std::vector<unsigned int>::iterator max_repeat = std::max_element(repeat_longest_range.begin(), repeat_longest_range.end());
	if (*max_repeat == 0) return;
	std::vector<std::vector<cv::Point2d>> connect_ranges;
	std::vector<cv::Point2d> connect_range;
	bool is_start = false;
	for (unsigned int i = 0; i < repeat_longest_range.size(); ++i){
		if (repeat_longest_range[i] == *max_repeat){
			is_start = true;
			connect_range.push_back(b_sample[i]);
		}
		else if (is_start){
			connect_ranges.push_back(connect_range);
			connect_range.clear();
			is_start = false;
		}
	}
	if (is_start) connect_ranges.push_back(connect_range);
	std::sort(connect_ranges.begin(), connect_ranges.end(), point2d_vector_size_cmp);

	double b_length = curve_length(connect_ranges[0]);
	double b_distance = cv::norm(connect_ranges[0][0] - connect_ranges[0][connect_ranges[0].size() - 1]);
	double b_ratio = b_length / b_distance;
	if ((a_ratio - a_ratio * 0.2 > b_ratio) ||
		(a_ratio + a_ratio * 0.4 < b_ratio)) return;

	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	for (unsigned int i = 1; i < connect_ranges[0].size(); ++i)
		cv::line(img_show,
		cv::Point2d(connect_ranges[0][i - 1].x * ps, connect_ranges[0][i - 1].y * -ps) + ds,
		cv::Point2d(connect_ranges[0][i].x * ps, connect_ranges[0][i].y * -ps) + ds,
		cv::Scalar(0, 0, 255));
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));

	return;
}

void mangaShow::compare_curves(){

	//for (unsigned int i = 0; i < curves.size(); ++i){
	//	printf("==> curve %u ~\n", i);
	//	std::reverse(match_curve.begin(), match_curve.end());
	//	compare_curve(match_curve, curves[i]);
	//	std::reverse(match_curve.begin(), match_curve.end());
	//	compare_curve(match_curve, curves[i]);
	//}

	building_pnt_to_curve();
	//std::vector<cv::Point2d> curve;
	//for (unsigned int i = 0; i < curves.size(); ++i){
	//	for (unsigned int j = 0; j < pnt_to_curve[curves[i][0]].size(); ++j){
	//		unsigned int cnnct_crv = pnt_to_curve[curves[i][0]][j];
	//		if (cnnct_crv <= i) continue;
	//		curve.clear();
	//		if (curves[cnnct_crv][0] == curves[i][0]) std::reverse(curves[cnnct_crv].begin(), curves[cnnct_crv].end());
	//		for (unsigned int k = 0; k < curves[cnnct_crv].size(); ++k) curve.push_back(curves[cnnct_crv][k]);
	//		for (unsigned int k = 1; k < curves[i].size(); ++k) curve.push_back(curves[i][k]);
	//		printf("==> curve %u X %u ~\n", i, cnnct_crv);
	//		std::reverse(match_curve.begin(), match_curve.end());
	//		compare_curve(match_curve, curve);
	//		std::reverse(match_curve.begin(), match_curve.end());
	//		compare_curve(match_curve, curve);
	//	}
	//	for (unsigned int j = 0; j < pnt_to_curve[curves[i][curves[i].size() - 1]].size(); ++j){
	//		unsigned int cnnct_crv = pnt_to_curve[curves[i][curves[i].size() - 1]][j];
	//		if (cnnct_crv <= i) continue;
	//		curve.clear();
	//		if (curves[cnnct_crv][curves[cnnct_crv].size() - 1] == curves[i][curves[i].size() - 1]) std::reverse(curves[cnnct_crv].begin(), curves[cnnct_crv].end());
	//		for (unsigned int k = 0; k < curves[i].size(); ++k) curve.push_back(curves[i][k]);
	//		for (unsigned int k = 1; k < curves[cnnct_crv].size(); ++k) curve.push_back(curves[cnnct_crv][k]);
	//		printf("==> curve %u X %u ~\n", i, cnnct_crv);
	//		std::reverse(match_curve.begin(), match_curve.end());
	//		compare_curve(match_curve, curve);
	//		std::reverse(match_curve.begin(), match_curve.end());
	//		compare_curve(match_curve, curve);
	//	}
	//}

	std::vector<cv::Point2d> curve;
	for (unsigned int i = 0; i < curves.size(); ++i){
		for (unsigned int j = 0; j < pnt_to_curve[curves[i][0]].size(); ++j){
			unsigned int a_crv = pnt_to_curve[curves[i][0]][j];
			if (a_crv == i) continue;
			curve.clear();
			if (curves[a_crv][0] == curves[i][0]) std::reverse(curves[a_crv].begin(), curves[a_crv].end());
			for (unsigned int k = 0; k < curves[a_crv].size(); ++k) curve.push_back(curves[a_crv][k]);
			for (unsigned int k = 1; k < curves[i].size(); ++k) curve.push_back(curves[i][k]);

			std::vector<cv::Point2d> curve_e;
			for (unsigned int k = 0; k < pnt_to_curve[curves[i][curves[i].size() - 1]].size(); ++k){
				unsigned int b_crv = pnt_to_curve[curves[i][curves[i].size() - 1]][k];
				if (b_crv == i) continue;
				curve_e = curve;
				if (curves[b_crv][curves[b_crv].size() - 1] == curves[i][curves[i].size() - 1]) std::reverse(curves[b_crv].begin(), curves[b_crv].end());
				for (unsigned int l = 1; l < curves[b_crv].size(); ++l) curve_e.push_back(curves[b_crv][l]);
				printf("==> curve %u X %u X %u ~\n", a_crv, i, b_crv);
				std::reverse(match_curve.begin(), match_curve.end());
				compare_curve(match_curve, curve_e);
				std::reverse(match_curve.begin(), match_curve.end());
				compare_curve(match_curve, curve_e);
			}
		}
	}

	return;
}

void mangaShow::draw_graph(){
	img_show = img_read.clone();
	
	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	for (const cv::Point2d &p : graph_pnts){
		for (const cv::Point2d &q : graph[p]){
			cv::line(img_show, cv::Point2d(p.x * ps, p.y * -ps) + ds, cv::Point2d(q.x * ps, q.y * -ps) + ds, green);
		}
	}

	for (const cv::Point2d &p : graph_pnts){
		cv::Scalar draw_color;
		switch (graph[p].size()){
			case 1: draw_color = red; break;
			case 2: draw_color = purple; break;
			case 3: draw_color = blue; break;
			case 4: draw_color = yellow; break;
			default: draw_color = cyan; break;
		}

		cv::circle(img_show, cv::Point2d(p.x * ps, p.y * -ps) + ds, 1, draw_color, CV_FILLED);
	}

	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/topology.png", canvas);

	printf("=> Drawing graph done.\n");
	return;
}

void mangaShow::rng_curves_color(){
	if (curves_color.size() != curves.size())
		curves_color.resize(curves.size());
	for (unsigned int i = 0; i < curves_color.size(); ++i)
		curves_color[i] = cv::Scalar(rng.uniform(0, 221), rng.uniform(0, 221), rng.uniform(0, 221));
	return;
}

void mangaShow::set_curves_drawable(int index, bool is_draw){
	if (curves_drawable.size() != curves.size()){
		curves_drawable.resize(curves.size());
		for (unsigned int i = 0; i < curves_drawable.size(); ++i)
			curves_drawable[i] = true;
	}

	if (index < 0 || index >= curves_drawable.size())
		for (unsigned int i = 0; i < curves_drawable.size(); ++i)
			curves_drawable[i] = is_draw;
	else curves_drawable[index] = is_draw;

	return;
}

void mangaShow::draw_curves(){
	if (curves_color.size() != curves.size()) rng_curves_color();
	if (curves_drawable.size() != curves.size()) set_curves_drawable();

	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	img_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < curves.size(); ++i)
		for (unsigned int j = 1; j < curves[i].size(); ++j)
			if (curves_drawable[i])
				cv::line(img_show,
				cv::Point2d(curves[i][j - 1].x * ps, curves[i][j - 1].y * -ps) + ds,
				cv::Point2d(curves[i][j].x * ps, curves[i][j].y * -ps) + ds,
				curves_color[i]);
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/curves.png", canvas);

	printf("=> Drawing curves done.\n");
	return;
}

bool mangaShow::is_read_img(){
	return img_read.dims ? true : false;
}

bool mangaShow::is_read_graph(){
	return graph.size() ? true : false;
}

bool mangaShow::is_read_match_curve(){
	return match_curve.size() ? true : false;
}

Bitmap ^mangaShow::mat2Bitmap(cv::Mat img){
	Bitmap ^curImage =
		img.channels() == 1 ?
		gcnew Bitmap(img.cols, img.rows, Imaging::PixelFormat::Format8bppIndexed) :
		gcnew Bitmap(img.cols, img.rows, Imaging::PixelFormat::Format24bppRgb);

	Imaging::BitmapData ^bitmapData = curImage->LockBits(System::Drawing::Rectangle(0, 0, curImage->Width, curImage->Height), Imaging::ImageLockMode::ReadWrite, curImage->PixelFormat);
	unsigned char *p = (unsigned char *)bitmapData->Scan0.ToPointer();
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.step; i++) {
			p[0] = img.data[i + j * img.step];
			++p;
		}
		p += bitmapData->Stride - img.step;
	}
	curImage->UnlockBits(bitmapData);
	return curImage;
}

Bitmap ^mangaShow::get_canvas_Bitmap(){
	return mat2Bitmap(canvas);
}

std::vector<bool> mangaShow::get_curves_drawable(){
	if (curves_drawable.size() != curves.size()) set_curves_drawable();
	return curves_drawable;
}

void mangaShow::test(){
	CurveDescriptor cd = CurveDescriptor(curves[0], 0.01, 2.0, true);
	std::vector<cv::Point2d> smooth_curve = cd.get_smooth_curve();
	std::vector<double> curvature = cd.get_curvature();
	std::vector<cv::Point2d> integration = cd.get_integration();
	std::vector<cv::Point2d> integration_sample = cd.get_integration_sample();
	std::vector<double> integration_curvature = cd.get_integration_curvature();

	draw_plot_graph(smooth_curve, "test_curve");
	std::vector<cv::Point2d> data;
	data.resize(curvature.size());
	for (unsigned int i = 0; i < curvature.size(); ++i){
		if (!i) data[i] = cv::Point2d(0, abs(curvature[i]));
		else data[i] = cv::Point2d(cv::norm(smooth_curve[i] - smooth_curve[i -1]) + data[i - 1].x, abs(curvature[i]));
	}
	draw_plot_graph(data, "test_curvature");
	draw_plot_graph(integration, "test_curve_integration");
	draw_plot_graph(integration_sample, "test_curve_integration_sample");
	data.resize(integration_curvature.size());
	for (unsigned int i = 0; i < integration_curvature.size(); ++i){
		data[i] = cv::Point2d(i, abs(integration_curvature[i]));
		printf("%lf, %lf\n", data[i].x, data[i].y);
	}
	draw_plot_graph(data, "test_curve_integration_curvature");
	b_itg_crvt = integration_curvature;
}

std::vector<cv::Point2d> mangaShow::link_curve(cv::Point2d p, cv::Point2d q, std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> &pnts_used_pnts){
	std::vector<cv::Point2d> curve;
	curve.push_back(p);
	pnts_used_pnts[p].erase(q);
	while (end_pnts.find(q) == end_pnts.end() &&
		junction_pnts.find(q) == junction_pnts.end()){
		curve.push_back(q);
		for (const cv::Point2d &r : graph[q]){
			if (p != r){
				p = q;
				q = r;
				break;
			}
		}
	}
	pnts_used_pnts[q].erase(curve[curve.size() - 1]);
	curve.push_back(q);
	return curve;
}

int mangaShow::normalize_cross_correlation(std::vector<double> a, std::vector<double> b){
	double max_v = 0;
	unsigned int offset = 0;

	double a_avg = 0;
	for (unsigned int i = 0; i < a.size(); ++i)
		a_avg += a[i];
	a_avg /= a.size();

	for (unsigned int i = 0; i < b.size() - a.size(); ++i){
		double b_avg = 0;
		for (unsigned int j = 0; j < a.size(); ++j)
			b_avg += b[i + j];
		b_avg /= a.size();

		double FT = 0, FF = 0, TT = 0;
		for (unsigned int j = 0; j < a.size(); ++j){
			double F = a[j] - a_avg;
			double T = b[i + j] - b_avg;
			FT += F * T;
			FF += F * F;
			TT += T * T;
		}
		double V = FT / sqrt(FF * TT);
		if (V > max_v){
			max_v = V;
			offset = i;
		}
	}
	printf("max_v: %lf\n", max_v);
	if (max_v < 0.5) return -1;
	return offset;
}

bool p_x_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.x < b.x; }
bool p_y_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.y < b.y; }
void mangaShow::draw_plot_graph(std::vector<cv::Point2d> data, char *win_name){
	int width = 1200, height = 900, padding = 50;
	cv::Mat plot_show = cv::Mat(height, width, CV_8UC3);
	std::vector<cv::Point2d>::iterator max_x = std::max_element(data.begin(), data.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator min_x = std::min_element(data.begin(), data.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator max_y = std::max_element(data.begin(), data.end(), p_y_cmp);
	std::vector<cv::Point2d>::iterator min_y = std::min_element(data.begin(), data.end(), p_y_cmp);
	
	double norm_x = (*max_x).x - (*min_x).x, norm_y = (*max_y).y - (*min_y).y;
	int inner_w = width - padding * 2, inner_h = height - padding * 2;
	for (unsigned int i = 1; i < data.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data[i - 1].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data[i].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data[i].y - (*min_y).y) / norm_y * inner_h)),
			cv::Scalar(200, 200, 0),
			2);
	}
	printf("max_y: %lf, min_y: %lf\n", (*max_y).y, (*min_y).y);
	cv::imshow(win_name, plot_show);
	cv::imwrite("results/plot.png", plot_show);
}

void mangaShow::draw_plot_graph(std::vector<cv::Point2d> data_a, std::vector<cv::Point2d> data_b, double offset, char *win_name){
	int width = 1200, height = 900, padding = 50;
	cv::Mat plot_show = cv::Mat(height, width, CV_8UC3);
	std::vector<cv::Point2d>::iterator max_x = std::max_element(data_b.begin(), data_b.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator min_x = std::min_element(data_b.begin(), data_b.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator max_y = std::max_element(data_b.begin(), data_b.end(), p_y_cmp);
	std::vector<cv::Point2d>::iterator min_y = std::min_element(data_b.begin(), data_b.end(), p_y_cmp);

	double norm_x = (*max_x).x - (*min_x).x, norm_y = (*max_y).y - (*min_y).y;
	int inner_w = width - padding * 2, inner_h = height - padding * 2;
	for (unsigned int i = 1; i < data_b.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data_b[i - 1].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_b[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data_b[i].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_b[i].y - (*min_y).y) / norm_y * inner_h)),
			cv::Scalar(200, 200, 0),
			2);
	}
	for (unsigned int i = 1; i < data_a.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data_a[i - 1].x + offset - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_a[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data_a[i].x + offset - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_a[i].y - (*min_y).y) / norm_y * inner_h)),
			cv::Scalar(0, 200, 200),
			2);
	}
	//printf("max_y: %lf, min_y: %lf\n", (*max_y).y, (*min_y).y);
	printf("=> Drawing %s plot done.\n", win_name);
	cv::imshow(win_name, plot_show);
	cv::imwrite("results/plot.png", plot_show);
}
