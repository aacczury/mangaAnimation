#include "mangaShow.h"

mangaShow::mangaShow(){
	img_read = img_show = canvas = cv::Mat();
	curves_color.clear();
	curves_drawable.clear();
}

void mangaShow::read_img(char *filename){
	img_read = cv::imread(filename, CV_LOAD_IMAGE_COLOR); // need exception handling
	img_show = img_read.clone();
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));

	QueryPerformanceFrequency(&freq);
	printf("=> Reading image done.\n");
}

void mangaShow::read_graph(char *filename, int g_s){
	if (g_s == MANGA_FACE)
		mangaFace = GraphFile(filename);
	else if (g_s == SAMPLE_FACE)
		sampleFace = GraphFile(filename);
	return;
}

void mangaShow::find_seed(){
	compare_curves(sampleFace.curves[8]);
	draw_sample_face(8);
	return;
}

void mangaShow::draw_graph(){
	img_show = img_read.clone();
	
	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	for (const cv::Point2d &p : mangaFace.graph_pnts){
		for (const cv::Point2d &q : mangaFace.graph[p]){
			cv::line(img_show, cv::Point2d(p.x * ps, p.y * -ps) + ds, cv::Point2d(q.x * ps, q.y * -ps) + ds, green);
		}
	}

	for (const cv::Point2d &p : mangaFace.graph_pnts){
		cv::Scalar draw_color;
		switch (mangaFace.graph[p].size()){
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
	if (curves_color.size() != mangaFace.curves.size()) curves_color.resize(mangaFace.curves.size());
	for (unsigned int i = 0; i < curves_color.size(); ++i)
		curves_color[i] = cv::Scalar(rng.uniform(0, 221), rng.uniform(0, 221), rng.uniform(0, 221));
	return;
}

void mangaShow::set_curves_drawable(int index, bool is_draw){
	if (curves_drawable.size() != mangaFace.curves.size()){ 
		curves_drawable.resize(mangaFace.curves.size());
		for (unsigned int i = 0; i < curves_drawable.size(); ++i) curves_drawable[i] = true;
	}

	if (index < 0 || index >= curves_drawable.size())
		for (unsigned int i = 0; i < curves_drawable.size(); ++i)
			curves_drawable[i] = is_draw;
	else curves_drawable[index] = is_draw;

	return;
}

void mangaShow::draw_curves(bool is_colorful){
	if (curves_color.size() != mangaFace.curves.size()) rng_curves_color();
	if (curves_drawable.size() != mangaFace.curves.size()) set_curves_drawable();

	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	img_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < mangaFace.curves.size(); ++i)
	for (unsigned int j = 1; j < mangaFace.curves[i].size(); ++j)
			if (curves_drawable[i])
				cv::line(img_show,
				cv::Point2d(mangaFace.curves[i][j - 1].x * ps, mangaFace.curves[i][j - 1].y * -ps) + ds,
				cv::Point2d(mangaFace.curves[i][j].x * ps, mangaFace.curves[i][j].y * -ps) + ds,
				is_colorful ? curves_color[i] : gray);
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/curves.png", canvas);

	printf("=> Drawing curves done.\n");
	return;
}

bool mangaShow::is_read_img(){
	return img_read.dims ? true : false;
}

bool mangaShow::is_read_mangaFace(){
	return mangaFace.graph.size() ? true : false;
}

bool mangaShow::is_read_sampleFace(){
	return sampleFace.graph.size() ? true : false;
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
	cv::imwrite("results/canvas.png", canvas);
	return mat2Bitmap(canvas);
}

Bitmap ^mangaShow::get_sample_canvas_Bitmap(){
	cv::imwrite("results/sample_canvas.png", sample_canvas);
	return mat2Bitmap(sample_canvas);
}

std::vector<bool> mangaShow::get_curves_drawable(){
	if (curves_drawable.size() != mangaFace.graph.size()) set_curves_drawable();
	return curves_drawable;
}

void mangaShow::test(){
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
	//printf("max_v: %lf\n", max_v);
	if (max_v < 0.7) return -1;
	return offset;
}

bool point2d_vector_size_cmp(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b){
	return a.size() > b.size();
}
void mangaShow::compare_curve(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b){
	double a_ratio = curve_length(a) / cv::norm(a[0] - a[a.size() - 1]);

	CurveDescriptor a_d = CurveDescriptor(a, 0.005, 3.0, true);
	if (a_d.is_error()) return;
	CurveDescriptor b_d = CurveDescriptor(b, 0.005, 3.0, true);
	if (b_d.is_error()) return;
	
	for (double i = 1; i <= 1.5; i += 0.5){
		if (a_d.curvature.size() / i <= 3) break;
		b_d.scaling_curvature(i);
		if (b_d.scale_curvature.size() < a_d.curvature.size()) continue;

		int offset = normalize_cross_correlation(a_d.curvature, b_d.scale_curvature);

		if (offset < 0) continue;
		std::vector<cv::Point2d> segment;
		for (unsigned int j = 0; j < a_d.curvature.size() / i; ++j)
			segment.push_back(b_d.curve[j + offset / i]);

		double b_length = curve_length(segment);
		double b_distance = cv::norm(segment[0] - segment[segment.size() - 1]);
		double b_ratio = b_length / b_distance;
		if ((a_ratio - a_ratio * 0.1 > b_ratio) ||
			(a_ratio + a_ratio * 0.2 < b_ratio)) continue;

		int ps = max(img_read.rows, img_read.cols);
		cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
		for (unsigned int i = 1; i < segment.size(); ++i)
			cv::line(img_show,
			cv::Point2d(segment[i - 1].x * ps, segment[i - 1].y * -ps) + ds,
			cv::Point2d(segment[i].x * ps, segment[i].y * -ps) + ds,
			green);
		cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
		break;
	}
	return;
}

void mangaShow::compare_curves(std::vector<cv::Point2d> sample_curve){

	for (unsigned int i = 0; i < mangaFace.curves.size(); ++i){
		std::reverse(sample_curve.begin(), sample_curve.end());
		compare_curve(sample_curve, mangaFace.curves[i]);
		std::reverse(sample_curve.begin(), sample_curve.end());
		compare_curve(sample_curve, mangaFace.curves[i]);
	}
	printf("==> curve 1 ~ done.\n");

	std::vector<cv::Point2d> curve;
	for (unsigned int i = 0; i < mangaFace.curves.size(); ++i){
		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.curves[i][0]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.curves[i][0]][j];
			if (a_crv <= i) continue;
			curve.clear();
			if (mangaFace.curves[a_crv][0] == mangaFace.curves[i][0])
				std::reverse(mangaFace.curves[a_crv].begin(), mangaFace.curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.curves[a_crv].size(); ++k) curve.push_back(mangaFace.curves[a_crv][k]);
			for (unsigned int k = 1; k < mangaFace.curves[i].size(); ++k) curve.push_back(mangaFace.curves[i][k]);
			std::reverse(sample_curve.begin(), sample_curve.end());
			compare_curve(sample_curve, curve);
			std::reverse(sample_curve.begin(), sample_curve.end());
			compare_curve(sample_curve, curve);
		}
		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.curves[i][mangaFace.curves[i].size() - 1]].size(); ++j){
			unsigned int cnnct_crv = mangaFace.pnt_to_curve[mangaFace.curves[i][mangaFace.curves[i].size() - 1]][j];
			if (cnnct_crv <= i) continue;
			curve.clear();
			if (mangaFace.curves[cnnct_crv][mangaFace.curves[cnnct_crv].size() - 1] == mangaFace.curves[i][mangaFace.curves[i].size() - 1])
				std::reverse(mangaFace.curves[cnnct_crv].begin(), mangaFace.curves[cnnct_crv].end());
			for (unsigned int k = 0; k < mangaFace.curves[i].size(); ++k) curve.push_back(mangaFace.curves[i][k]);
			for (unsigned int k = 1; k < mangaFace.curves[cnnct_crv].size(); ++k) curve.push_back(mangaFace.curves[cnnct_crv][k]);
			std::reverse(sample_curve.begin(), sample_curve.end());
			compare_curve(sample_curve, curve);
			std::reverse(sample_curve.begin(), sample_curve.end());
			compare_curve(sample_curve, curve);
		}
	}
	printf("==> curve 2 ~ done.\n");

	for (unsigned int i = 0; i < mangaFace.curves.size(); ++i){
		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.curves[i][0]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.curves[i][0]][j];
			if (a_crv == i) continue;
			curve.clear();
			if (mangaFace.curves[a_crv][0] == mangaFace.curves[i][0])
				std::reverse(mangaFace.curves[a_crv].begin(), mangaFace.curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.curves[a_crv].size(); ++k) curve.push_back(mangaFace.curves[a_crv][k]);
			for (unsigned int k = 1; k < mangaFace.curves[i].size(); ++k) curve.push_back(mangaFace.curves[i][k]);
			
			std::vector<cv::Point2d> curve_e;
			for (unsigned int k = 0; k < mangaFace.pnt_to_curve[mangaFace.curves[i][mangaFace.curves[i].size() - 1]].size(); ++k){
				unsigned int b_crv = mangaFace.pnt_to_curve[mangaFace.curves[i][mangaFace.curves[i].size() - 1]][k];
				if (b_crv == i) continue;
				curve_e = curve;
				if (mangaFace.curves[b_crv][mangaFace.curves[b_crv].size() - 1] == mangaFace.curves[i][mangaFace.curves[i].size() - 1])
					std::reverse(mangaFace.curves[b_crv].begin(), mangaFace.curves[b_crv].end());
				for (unsigned int l = 1; l < mangaFace.curves[b_crv].size(); ++l) curve_e.push_back(mangaFace.curves[b_crv][l]);
				std::reverse(sample_curve.begin(), sample_curve.end());
				compare_curve(sample_curve, curve_e);
				std::reverse(sample_curve.begin(), sample_curve.end());
				compare_curve(sample_curve, curve_e);
			}
		}
	}
	printf("==> curve 3 ~ done.\n");

	return;
}

double mangaShow::curve_length(std::vector<cv::Point2d> curve){
	double total = 0;
	for (unsigned int i = 1; i < curve.size(); ++i)	total += cv::norm(curve[i] - curve[i - 1]);
	return total;
}

template<typename T>
T &mangaShow::ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c){
	return ((T *)m.data)[(p.y * m.cols + p.x) * m.channels() + c];
}

void mangaShow::draw_sample_face(unsigned int sample){
	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	cv::Mat sample_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < sampleFace.curves.size(); ++i)
		for (unsigned int j = 1; j < sampleFace.curves[i].size(); ++j)
			cv::line(sample_show,
			cv::Point2d(sampleFace.curves[i][j - 1].x * ps, sampleFace.curves[i][j - 1].y * -ps) + ds,
			cv::Point2d(sampleFace.curves[i][j].x * ps, sampleFace.curves[i][j].y * -ps) + ds,
			i == sample ? green : gray);
	cv::resize(sample_show, sample_canvas, cv::Size(sample_show.cols * scale, sample_show.rows * scale));
	cv::imwrite("results/sample_canvas.png", sample_canvas);
	return;
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
			cyan,
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
			cyan,
			2);
	}
	for (unsigned int i = 1; i < data_a.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data_a[i - 1].x + offset - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_a[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data_a[i].x + offset - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_a[i].y - (*min_y).y) / norm_y * inner_h)),
			yellow,
			2);
	}
	//printf("max_y: %lf, min_y: %lf\n", (*max_y).y, (*min_y).y);
	printf("=> Drawing %s plot done.\n", win_name);
	cv::imshow(win_name, plot_show);
	cv::imwrite("results/plot.png", plot_show);
}