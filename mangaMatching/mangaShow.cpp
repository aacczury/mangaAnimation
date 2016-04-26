#include "mangaShow.h"

bool point2d_vector_size_cmp(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b){ return a.size() > b.size(); }
bool p_x_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.x < b.x; }
bool p_y_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.y < b.y; }

mangaShow::mangaShow(){
	img_read = img_show = canvas = cv::Mat();
	curves_color.clear();
	curves_drawable.clear();
	mangaFace_CD.clear();
	sampleFace_CD.clear();
}

void mangaShow::read_img(char *filename){
	img_read = cv::imread(filename, CV_LOAD_IMAGE_COLOR); // need exception handling
	img_show = img_read.clone();
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));

	QueryPerformanceFrequency(&freq);
	printf("=> Reading image done.\n");
}

void mangaShow::read_graph(char *filename, int g_s){
	if (g_s == MANGA_FACE){
		mangaFace = GraphFile(filename);
		for (unsigned int i = 0; i < mangaFace.curves.size(); ++i){
			mangaFace_CD.push_back(CurveDescriptor(mangaFace.curves[i], 0.005, 3.0, true));
			mangaFace.sample_curves.push_back(mangaFace_CD[i].curve);
		}
	}
	else if (g_s == SAMPLE_FACE){
		sampleFace = GraphFile(filename);
		for (unsigned int i = 0; i < sampleFace.curves.size(); ++i){
			sampleFace_CD.push_back(CurveDescriptor(sampleFace.curves[i], 0.005, 3.0, true));
			sampleFace.sample_curves.push_back(sampleFace_CD[i].curve);
		}
	}
	return;
}

void mangaShow::find_seed(){
	rng_curves_color();
	sample_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	seeds.clear(), seeds.resize(sampleFace.sample_curves.size());
	//for (unsigned int i = 0; i < sampleFace.sample_curves.size(); ++i){
	//	compare_curves_with_primitive(sampleFace.curves[i], i);
	//	draw_sample_face(i, curves_color[i]);
	//}
	compare_curves_with_primitive(sampleFace.sample_curves[0], 0);
	//draw_sample_face(0, curves_color[0]);
	compare_curves_with_primitive(sampleFace.sample_curves[1], 1);
	//draw_sample_face(1, curves_color[1]);
	
	std::vector<double> a_r_a = calculate_relative_angles(sampleFace_CD[0], sampleFace_CD[1]);

	double min_a = 1 * 2;
	unsigned int min_i = 0, min_j = 0;
	for (unsigned int i = 0; i < seeds[0].size(); ++i){
		CurveDescriptor a_d = CurveDescriptor(seeds[0][i], 0.005, 3.0, true);
		if (a_d.is_error()) continue;
		for (unsigned int j = 0; j < seeds[1].size(); ++j){
			CurveDescriptor b_d = CurveDescriptor(seeds[1][j], 0.005, 3.0, true);
			if (b_d.is_error()) continue;

			std::vector<double> b_r_a = calculate_relative_angles(a_d, b_d);

			double diff = 0;
			for (unsigned int k = 0; k < 2; ++k)
				diff += abs(a_r_a[k] - b_r_a[k]);
			if (diff < min_a) min_a = diff, min_i = i, min_j = j;
		}
	}

	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	for (unsigned int i = 1; i < sampleFace.sample_curves[0].size(); ++i)
		cv::line(sample_show,
		cv::Point2d(sampleFace.sample_curves[0][i - 1].x * ps, sampleFace.sample_curves[0][i - 1].y * -ps) + ds,
		cv::Point2d(sampleFace.sample_curves[0][i].x * ps, sampleFace.sample_curves[0][i].y * -ps) + ds,
		green);

	for (unsigned int j = 1; j < seeds[0][min_i].size(); ++j){
		cv::line(img_show,
			cv::Point2d(seeds[0][min_i][j - 1].x * ps, seeds[0][min_i][j - 1].y * -ps) + ds,
			cv::Point2d(seeds[0][min_i][j].x * ps, seeds[0][min_i][j].y * -ps) + ds,
			green);
	}

	for (unsigned int i = 1; i < sampleFace.sample_curves[1].size(); ++i)
		cv::line(sample_show,
		cv::Point2d(sampleFace.sample_curves[1][i - 1].x * ps, sampleFace.sample_curves[1][i - 1].y * -ps) + ds,
		cv::Point2d(sampleFace.sample_curves[1][i].x * ps, sampleFace.sample_curves[1][i].y * -ps) + ds,
		blue);
	for (unsigned int j = 1; j < seeds[1][min_j].size(); ++j){
		cv::line(img_show,
			cv::Point2d(seeds[1][min_j][j - 1].x * ps, seeds[1][min_j][j - 1].y * -ps) + ds,
			cv::Point2d(seeds[1][min_j][j].x * ps, seeds[1][min_j][j].y * -ps) + ds,
			blue);
	}

	cv::resize(sample_show, sample_canvas, cv::Size(sample_show.cols * scale, sample_show.rows * scale));
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
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
		curves_color[i] = cv::Scalar(rng.uniform(50, 200), rng.uniform(50, 200), rng.uniform(50, 200));
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
	if (curves_color.size() != mangaFace.sample_curves.size()) rng_curves_color();
	if (curves_drawable.size() != mangaFace.sample_curves.size()) set_curves_drawable();

	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	img_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i)
	for (unsigned int j = 1; j < mangaFace.sample_curves[i].size(); ++j)
			if (curves_drawable[i])
				cv::line(img_show,
				cv::Point2d(mangaFace.sample_curves[i][j - 1].x * ps, mangaFace.sample_curves[i][j - 1].y * -ps) + ds,
				cv::Point2d(mangaFace.sample_curves[i][j].x * ps, mangaFace.sample_curves[i][j].y * -ps) + ds,
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
	unsigned int notable_index = max_curvature_index(sampleFace_CD[1].curvature);
	cv::Point2d t = caculate_tangent(sampleFace_CD[1].curve, notable_index);
	cv::Point2d p = sampleFace_CD[1].curve[notable_index];
	double c = p.x * t.y + p.y * -t.x;
	std::vector<cv::Point2d> data_a;
	std::vector<cv::Point2d>::iterator min_it = std::min_element(sampleFace_CD[1].curve.begin(), sampleFace_CD[1].curve.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator max_it = std::max_element(sampleFace_CD[1].curve.begin(), sampleFace_CD[1].curve.end(), p_x_cmp);
	data_a.push_back(cv::Point2d((*min_it).x, (c - (*min_it).x * t.y) / -t.x));
	data_a.push_back(cv::Point2d((*max_it).x, (c - (*max_it).x * t.y) / -t.x));
	draw_plot_graph(data_a, sampleFace_CD[1].curve, 0, "QQ");
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
	if (max_v < 0.8) return -1;
	return offset;
}

std::vector<cv::Point2d> mangaShow::compare_curve(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b){
	std::vector<cv::Point2d> segment;

	double a_ratio = curve_length(a) / cv::norm(a[0] - a[a.size() - 1]);

	CurveDescriptor a_d = CurveDescriptor(a, 3.0, true);
	if (a_d.is_error()) return std::vector<cv::Point2d>();
	CurveDescriptor b_d = CurveDescriptor(b, 3.0, true);
	if (b_d.is_error()) return std::vector<cv::Point2d>();

	unsigned int a_notable_index = max_curvature_index(a_d.curvature);
	double a_degree = abc_degree(a[0], a[a_notable_index], a[a.size() - 1]);

	for (double i = 1; i <= 1.5; i += 0.5){
		segment.clear();
		if (a_d.curvature.size() / i <= 3) return std::vector<cv::Point2d>();
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

		CurveDescriptor s_d = CurveDescriptor(segment, 3.0, true);
		if (s_d.is_error()) continue;
		unsigned int b_notable_index = max_curvature_index(s_d.curvature);
		double b_degree = abc_degree(segment[0], segment[b_notable_index], segment[segment.size() - 1]);
		if ((a_degree - CV_PI / 8 > b_degree) ||
			(a_degree + CV_PI / 8 < b_degree)) continue;

		return segment;
	}
	return std::vector<cv::Point2d>();
}

void mangaShow::compare_curve_add_seed(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b, unsigned int p_i){
	std::vector<cv::Point2d> seed;

	std::reverse(a.begin(), a.end());
	seed = compare_curve(a, b);
	if (seed.size() > 0) seeds[p_i].push_back(seed);

	std::reverse(a.begin(), a.end());
	seed = compare_curve(a, b);
	if (seed.size() > 0) seeds[p_i].push_back(seed);

	return;
}

void mangaShow::compare_curves_with_primitive(std::vector<cv::Point2d> sample_curve, unsigned int p_i){
	std::vector<bool> curve_is_visited;
	curve_is_visited.resize(mangaFace.sample_curves.size()); // all will be zero


	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
		if (mangaFace.sample_curves[i].size() == 0) continue;

		std::vector<cv::Point2d> curve;
		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]][j];
			if (a_crv == i || mangaFace.sample_curves[a_crv].size() == 0) continue;
			curve.clear();

			if (mangaFace.sample_curves[a_crv][0] == mangaFace.sample_curves[i][0])
				std::reverse(mangaFace.sample_curves[a_crv].begin(), mangaFace.sample_curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.sample_curves[a_crv].size(); ++k) curve.push_back(mangaFace.sample_curves[a_crv][k]);
			for (unsigned int k = 1; k < mangaFace.sample_curves[i].size(); ++k) curve.push_back(mangaFace.sample_curves[i][k]);

			std::vector<cv::Point2d> curve_e;
			for (unsigned int k = 0; k < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]].size(); ++k){
				unsigned int b_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]][k];
				if (b_crv == i || mangaFace.sample_curves[b_crv].size() == 0) continue;
				curve_e = curve;

				if (mangaFace.sample_curves[b_crv][mangaFace.sample_curves[b_crv].size() - 1] == mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1])
					std::reverse(mangaFace.sample_curves[b_crv].begin(), mangaFace.sample_curves[b_crv].end());
				for (unsigned int l = 1; l < mangaFace.sample_curves[b_crv].size(); ++l) curve_e.push_back(mangaFace.sample_curves[b_crv][l]);
				
				compare_curve_add_seed(sample_curve, curve_e, p_i);
				curve_is_visited[a_crv] = curve_is_visited[i] = curve_is_visited[b_crv] = true;
			}
		}
	}
	printf("==> curve 3 ~ done.\n");

	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
		if (curve_is_visited[i] || mangaFace.sample_curves[i].size() == 0) continue;

		std::vector<cv::Point2d> curve;
		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]][j];
			if (a_crv <= i || curve_is_visited[a_crv] || mangaFace.sample_curves[a_crv].size() == 0) continue;
			curve.clear();

			if (mangaFace.sample_curves[a_crv][0] == mangaFace.sample_curves[i][0])
				std::reverse(mangaFace.sample_curves[a_crv].begin(), mangaFace.sample_curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.sample_curves[a_crv].size(); ++k) curve.push_back(mangaFace.sample_curves[a_crv][k]);
			for (unsigned int k = 1; k < mangaFace.sample_curves[i].size(); ++k) curve.push_back(mangaFace.sample_curves[i][k]);

			compare_curve_add_seed(sample_curve, curve, p_i);
			curve_is_visited[a_crv] = curve_is_visited[i] = true;
		}

		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]][j];
			if (a_crv <= i || curve_is_visited[a_crv] || mangaFace.sample_curves[a_crv].size() == 0) continue;
			curve.clear();

			if (mangaFace.sample_curves[a_crv][mangaFace.sample_curves[a_crv].size() - 1] == mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1])
				std::reverse(mangaFace.sample_curves[a_crv].begin(), mangaFace.sample_curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.sample_curves[i].size(); ++k) curve.push_back(mangaFace.sample_curves[i][k]);
			for (unsigned int k = 1; k < mangaFace.sample_curves[a_crv].size(); ++k) curve.push_back(mangaFace.sample_curves[a_crv][k]);

			compare_curve_add_seed(sample_curve, curve, p_i);
			curve_is_visited[a_crv] = curve_is_visited[i] = true;
		}
	}
	printf("==> curve 2 ~ done.\n");

	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
		if (curve_is_visited[i] || mangaFace.sample_curves[i].size() == 0) continue;

		compare_curve_add_seed(sample_curve, mangaFace.sample_curves[i], p_i);
		curve_is_visited[i] = true;
	}
	printf("==> curve 1 ~ done.\n");

	return;
}

std::vector<double> mangaShow::calculate_relative_angles(CurveDescriptor a, CurveDescriptor b){
	std::vector<double> relative_angles;
	relative_angles.resize(2);

	unsigned int a_index = max_curvature_index(a.curvature);
	unsigned int b_index = max_curvature_index(b.curvature);
	cv::Point2d t1 = caculate_tangent(a.curve, a_index);
	cv::Point2d t2 = caculate_tangent(b.curve, b_index);
	cv::Point2d n1 = b.curve[b_index] - a.curve[a_index];
	cv::Point2d n2 = -n1;

	relative_angles[0] = (n1.cross(t1) >= 0 ? v_degree(n1, t1) : v_degree(n1, -t1)) / CV_PI;
	relative_angles[1] = (n2.cross(t2) >= 0 ? v_degree(n2, t2) : v_degree(n2, -t2)) / CV_PI;

	return relative_angles;
}

template<typename T>
T &mangaShow::ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c){
	return ((T *)m.data)[(p.y * m.cols + p.x) * m.channels() + c];
}

double mangaShow::curve_length(std::vector<cv::Point2d> curve){
	double total = 0;
	for (unsigned int i = 1; i < curve.size(); ++i)	total += cv::norm(curve[i] - curve[i - 1]);
	return total;
}

// p to p1<->p2 line distance
double mangaShow::perpendicular_distance(cv::Point2d p, cv::Point2d p1, cv::Point2d p2){
	cv::Point2d v1 = p - p1, v2 = p2 - p1;
	return std::fabs(v1.cross(v2)) / cv::norm(p1 - p2);
}

// only need input line & max_depth; p, q, depth using on recursive;
std::vector<unsigned int> mangaShow::douglas_peucker(std::vector<cv::Point2d> &line, int max_depth, int p, int q, int depth){
	if (q == -1) q = line.size();
	if (p + 1 >= q - 1 || depth >= max_depth) return std::vector<unsigned int>();

	double max_dis = 0;
	int max_index = -1;
	std::vector<unsigned int> angle_index, sub_angle_index;
	for (size_t i = p + 1; i < q - 1; ++i){
		double dis = perpendicular_distance(line[i], line[p], line[q - 1]);
		if (dis > max_dis){
			max_dis = dis;
			max_index = i;
		}
	}
	if (max_index == -1) return std::vector<unsigned int>();
	angle_index.push_back(max_index);
	sub_angle_index = douglas_peucker(line, max_depth, p, (p + q) >> 1, depth + 1);
	for (size_t i = 0; i < sub_angle_index.size(); ++i) angle_index.push_back(sub_angle_index[i]);
	sub_angle_index = douglas_peucker(line, max_depth, (p + q) >> 1, q, depth + 1);
	for (size_t i = 0; i < sub_angle_index.size(); ++i) angle_index.push_back(sub_angle_index[i]);

	if (p == 0 && q == line.size()){ // 最表層將端點加入
		angle_index.push_back(p);
		angle_index.push_back(q - 1);
	}
	// 去掉重複的點
	std::sort(angle_index.begin(), angle_index.end());
	std::vector<unsigned int>::iterator it = std::unique(angle_index.begin(), angle_index.end());
	angle_index.resize(std::distance(angle_index.begin(), it));

	return angle_index;
}

unsigned int mangaShow::max_curvature_index(std::vector<double> curvature){
	double max_curvature = 0;
	unsigned int max_index = 0;
	for (unsigned int i = 0; i < curvature.size(); ++i){
		if (curvature[i] > max_curvature){
			max_curvature = curvature[i];
			max_index = i;
		}
	}
	return max_index;
}

// return [0, PI]
double mangaShow::v_degree(cv::Point2d v1, cv::Point2d v2){
	return acos(v1.ddot(v2) / cv::norm(v1) / cv::norm(v2));
}

// return [0, PI]
double mangaShow::abc_degree(cv::Point2d a, cv::Point2d b, cv::Point2d c){
	cv::Point2d v1 = a - b, v2 = c - b;
	return v_degree(v1, v2);
}

cv::Point2d mangaShow::caculate_tangent(std::vector<cv::Point2d> curve, unsigned int index){
	if (index < 2 || index >= curve.size() - 2){
		printf("... Can't caculate tangent\n");
		return cv::Point2d(0, 0);
	}
	cv::Point2d v0 = curve[index - 1] - curve[index - 2];
	cv::Point2d v1 = curve[index] - curve[index - 1];
	cv::Point2d v2 = curve[index + 1] - curve[index];
	cv::Point2d v3 = curve[index + 2] - curve[index + 1];
	cv::Point2d t = v0 * 1 + v1 * 3 + v2 * 3 + v3 * 1;
	return cv::Point2d(t.x / 8, t.y / 8);
}

void mangaShow::draw_sample_face(unsigned int sample, cv::Scalar color){
	int ps = max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	for (unsigned int i = 1; i < sampleFace.sample_curves[sample].size(); ++i)
			cv::line(sample_show,
			cv::Point2d(sampleFace.sample_curves[sample][i - 1].x * ps, sampleFace.sample_curves[sample][i - 1].y * -ps) + ds,
			cv::Point2d(sampleFace.sample_curves[sample][i].x * ps, sampleFace.sample_curves[sample][i].y * -ps) + ds,
			color);
	cv::resize(sample_show, sample_canvas, cv::Size(sample_show.cols * scale, sample_show.rows * scale));

	for (unsigned int i = 0; i < seeds[sample].size(); ++i){
		for (unsigned int j = 1; j < seeds[sample][i].size(); ++j){
				cv::line(img_show,
					cv::Point2d(seeds[sample][i][j - 1].x * ps, seeds[sample][i][j - 1].y * -ps) + ds,
					cv::Point2d(seeds[sample][i][j].x * ps, seeds[sample][i][j].y * -ps) + ds,
					color);
		}
	}
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));

	cv::imwrite("results/sample_canvas.png", sample_canvas);
	return;
}

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