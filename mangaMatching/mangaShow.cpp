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
	
	for (const cv::Point2d &p : end_pnts){
		cv::Point2d q = *(graph[p].begin());
		if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
		curves.push_back(link_curve(p, q, pnts_used_pnts));
	}

	for (const cv::Point2d &p : junction_pnts){
		for (const cv::Point2d &q : graph[p]){
			if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
			curves.push_back(link_curve(p, q, pnts_used_pnts));
		}
	}

	std::sort(curves.begin(), curves.end(), curve_length_cmp);
	printf("=> Building curves done\n");
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

void mangaShow::compare_curves(){
	std::vector<cv::Point2d> r_curve = rotate_curve(match_curve);
	//draw_plot_graph(r_curve, "r_curve");
	//std::reverse(match_curve.begin(), match_curve.end());
	//CurveDescriptor a_c = CurveDescriptor(match_curve, 0.01, 3.0, true);
	//std::vector<double> a_curvature = a_c.get_curvature();
	//for (unsigned int i = 0; i < a_curvature.size(); ++i) a_curvature[i] = abs(a_curvature[i]);
	//printf("==> a curvature size: %u\n", a_curvature.size());
	//a_itg_crvt = a_c.get_integration_curvature();
	//std::vector<cv::Point2d> data_a;
	//data_a.resize(a_curvature.size());
	//for (unsigned int i = 0; i < a_curvature.size(); ++i) data_a[i] = cv::Point2d(i, abs(a_curvature[i]));
	//
	//std::vector<cv::Point2d> curve;
	//if (curves[1][0] == curves[4][0] || curves[1][0] == curves[4][curves[4].size() - 1])
	//	std::reverse(curves[1].begin(), curves[1].end());
	//if (curves[4][curves[4].size() - 1] == curves[1][curves[1].size() - 1])
	//	std::reverse(curves[4].begin(), curves[4].end());
	//for (unsigned int i = 0; i < curves[1].size(); ++i) curve.push_back(curves[1][i]);
	//for (unsigned int i = 0; i < curves[4].size(); ++i) curve.push_back(curves[4][i]);
	//CurveDescriptor b_c = CurveDescriptor(curve, 0.005, 3.0, true);
	//std::vector<double> b_tmp_curvature = b_c.get_curvature();
	//std::vector<double> b_curvature;
	//b_curvature.resize(b_tmp_curvature.size() * 2);
	//for (unsigned int i = 0; i < b_curvature.size(); ++i){
	//	if (!i) b_curvature[i] = abs(b_tmp_curvature[i / 2]) / 2;
	//	else if (i % 2) b_curvature[i] = abs(b_tmp_curvature[i / 2]);
	//	else if (!(i % 2))b_curvature[i] = (abs(b_tmp_curvature[i / 2 - 1]) + abs(b_tmp_curvature[i / 2])) / 2;
	//}
	//printf("==> b curvature size: %u\n", b_curvature.size());
	//b_itg_crvt = b_c.get_integration_curvature();
	//std::vector<cv::Point2d> data_b;
	//data_b.resize(b_curvature.size());
	//for (unsigned int i = 0; i < b_curvature.size(); ++i) data_b[i] = cv::Point2d(i, abs(b_curvature[i]));
	//
	//unsigned int offset = normalize_cross_correlation(a_curvature, b_curvature);
	//draw_plot_graph(data_a, data_b, offset, "compare intg crvt");
	////std::vector<cv::Point2d> segment = b_c.get_segment_curve(offset, offset + a_itg_crvt.size());
	////
	//std::vector<cv::Point2d> sample_b = b_c.get_sample_curve();
	//std::vector<cv::Point2d> segment;
	//for (unsigned int i = 0; i < a_curvature.size() / 2; ++i)
	//	segment.push_back(sample_b[offset / 2 + i]);
	//int ps = max(img_read.rows, img_read.cols);
	//cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	//for (unsigned int i = 1; i < segment.size(); ++i)
	//	cv::line(img_show,
	//	cv::Point2d(segment[i - 1].x * ps, segment[i - 1].y * -ps) + ds,
	//	cv::Point2d(segment[i].x * ps, segment[i].y * -ps) + ds,
	//	cv::Scalar(0, 0, 255));
	//cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	//return;
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

unsigned int mangaShow::normalize_cross_correlation(std::vector<double> a, std::vector<double> b){
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

// m, c. y = mx + c
cv::Point2d mangaShow::linear_regression(std::vector<cv::Point2d> pnts){
	double avg_x = 0, avg_y = 0;
	for (unsigned int i = 0; i < pnts.size(); ++i){
		avg_x += pnts[i].x;
		avg_y += pnts[i].y;
	}
	avg_x /= pnts.size();
	avg_y /= pnts.size();
	double S_xy = 0, S_xx = 0;
	for (unsigned int i = 0; i < pnts.size(); ++i){
		S_xy += (pnts[i].x - avg_x) * (pnts[i].y - avg_y);
		S_xx += (pnts[i].x - avg_x) * (pnts[i].x - avg_x);
	}
	double m = S_xy / S_xx;
	double c = avg_y - m * avg_x;
	return cv::Point2d(m, c);
}

// mx -  y + c = 0
// ax + by + c = 0 => |ax + by + c| / (a^2 + b^2)^0.5
double mangaShow::pnt_to_line_length(cv::Point2d pnt, cv::Point2d mc){
	double m = mc.x, c = mc.y;
	return abs(m * pnt.x - pnt.y + c) / sqrt(m * m + 1);
}

// mx - y + c  = 0
// mx + y + c' = 0
cv::Point2d mangaShow::project_pnt(cv::Point2d pnt, cv::Point2d mc){
	double m = mc.x, c = mc.y;
	double _c = -m * pnt.x - pnt.y;
	double x = (_c + c) / 2 / m;
	double y = (_c - c) / 2;
	return cv::Point2d(x, y);
}

std::vector<cv::Point2d> mangaShow::rotate_curve(std::vector<cv::Point2d> curve){
	std::reverse(curve.begin(), curve.end());
	for (unsigned int i = 0; i < curve.size(); ++i) printf("%lf, %lf\n", curve[i].x, curve[i].y);
	cv::Point2d mc = linear_regression(curve);
	std::vector<cv::Point2d> regline;
	regline.push_back(cv::Point2d(curve[0].x, mc.x * curve[0].x + mc.y));
	regline.push_back(cv::Point2d(curve[curve.size() - 1].x, mc.x * curve[curve.size() - 1].x + mc.y));
	draw_plot_graph(regline, curve, 0, "QQ");
	std::vector<cv::Point2d> r_curve;
	r_curve.resize(curve.size());

	cv::Point2d propnt0 = project_pnt(curve[0], mc);
	cv::Point2d v1 = cv::Point2d(1, mc.x);
	cv::Point2d n1 = curve[0] - propnt0;
	for (unsigned int i = 0; i < curve.size(); ++i){
		cv::Point2d propnt = project_pnt(curve[i], mc);
		printf("%lf, %lf\n", propnt.x, propnt.y);
		double x_dir = v1.ddot(propnt - propnt0) >= 0 ? 1 : -1;
		double y_dir = n1.ddot(curve[i] - propnt) >= 0 ? 1 : -1;
		r_curve[i] = cv::Point2d(cv::norm(propnt - propnt0),
			cv::norm(curve[i] - propnt) * y_dir);
		//printf("%lf, %lf\n", r_curve[i].x, r_curve[i].y);
	}
	draw_plot_graph(r_curve, "QQQ");
	return r_curve;
}