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

	printf("=> Building curves done\n");
	return;
}

void mangaShow::caculate_curve(){
	std::vector<cv::Point2d> curve;
	for (unsigned int i = 0; i < curves[0].size(); ++i)
		curve.push_back(curves[0][i]);

	CurveDescriptor cd = CurveDescriptor(curve, curve.size() / 5, 3.0, true);
	curve = cd.get_smooth_curve();
	std::vector<double> curvature = cd.get_curvature();
	std::vector<double> curvature_integration = cd.get_curvature_integration();

	for (unsigned int i = 1; i < curve.size(); ++i){
		cv::line(img_show, cv::Point(round(curve[i].x), round(curve[i].y)), cv::Point(round(curve[i - 1].x), round(curve[i - 1].y)), cv::Scalar(0, 0, 255));
	}
	cv::resize(img_show, canvas, cv::Size(img_show.cols * 3, img_show.rows * 3));

	std::vector<cv::Point2d> data;
	data.resize(curve.size());
	for (unsigned int i = 0; i < curve.size() - 1; ++i){
		if (i == 0) data[i] = cv::Point2d(0, abs(curvature_integration[i]));
		else data[i] = cv::Point2d(data[i - 1].x + cv::norm(curve[i] - curve[i - 1]), abs(curvature_integration[i]));
		printf("%lf\n", curvature_integration[i]);
	}
	draw_plot_graph(data);
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
	if (img_read.dims) return true;
	return false;
}

bool mangaShow::is_read_graph(){
	if (graph.size()) return true;
	return false;
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
void mangaShow::draw_plot_graph(std::vector<cv::Point2d> data){
	int width = 1200, height = 900, padding = 50;
	cv::Mat plot_show = cv::Mat(height, width, CV_8UC3);
	std::sort(data.begin(), data.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator max_y = std::max_element(data.begin(), data.end(), p_y_cmp);
	std::vector<cv::Point2d>::iterator min_y = std::min_element(data.begin(), data.end(), p_y_cmp);
	
	double norm_x = data[data.size() - 1].x - data[0].x, norm_y = (*max_y).y - (*min_y).y;
	int inner_w = width - padding * 2, inner_h = height - padding * 2;
	for (unsigned int i = 1; i < data.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data[i - 1].x - data[0].x) / norm_x * inner_w), padding + inner_h - round((data[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data[i].x - data[0].x) / norm_x * inner_w), padding + inner_h - round((data[i].y - (*min_y).y) / norm_y * inner_h)),
			cv::Scalar(200, 200, 0),
			2);
	}
	printf("max_y: %lf, min_y: %lf\n", (*max_y).y, (*min_y).y);
	cv::imshow("plot", plot_show);
	cv::imwrite("results/plot.png", plot_show);
}