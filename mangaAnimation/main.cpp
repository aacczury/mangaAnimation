#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <ilcplex/ilocplex.h>
ILOSTLBEGIN

char ostr[200];

std::vector<cv::Mat> imgs;
std::vector<std::vector<cv::Point2f>> notable_pnts;

template<typename T>
T &ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c){
	return ((T *)m.data)[(p.y * m.cols + p.x) * m.channels() + c];
}


cv::Point2f get_midpoint(std::vector<cv::Point2f> &pnts){
	cv::Point2f avg_pnt(0, 0);
	for (const cv::Point2f &p : pnts) avg_pnt += p;
	avg_pnt.x /= pnts.size();
	avg_pnt.y /= pnts.size();

	return avg_pnt;
}

cv::Point2f get_d(std::vector<cv::Point2f> &pnt){
	cv::Point2f mbc = pnt[1] + pnt[2];
	mbc.x /= 2, mbc.y /= 2;
	return mbc + mbc - pnt[0];
}

void using_camera_calibration(std::vector<cv::Point2f> &a_notable_pnts, std::vector<cv::Point2f> &b_notable_pnts, char *filename){
	std::vector<cv::Point2f> a_tri, b_tri;

	a_tri.push_back(a_notable_pnts[0]), a_tri.push_back(a_notable_pnts[5]), a_tri.push_back(a_notable_pnts[6]);
	b_tri.push_back(b_notable_pnts[0]), b_tri.push_back(b_notable_pnts[5]), b_tri.push_back(b_notable_pnts[6]);

	cv::Point2f a_mid = get_midpoint(a_tri);
	cv::Point2f b_mid = get_midpoint(b_tri);

	for (cv::Point2f &p : a_tri) p -= a_mid;
	for (cv::Point2f &p : b_tri) p -= b_mid;

	a_tri.push_back(get_d(a_tri));
	b_tri.push_back(get_d(b_tri));

	std::vector<std::vector<cv::Point3f>> obj;
	std::vector<std::vector<cv::Point2f>> img;

	std::vector<cv::Point3f> o;
	for (cv::Point2f &p : a_tri) o.push_back(cv::Point3f(p.x, p.y, 0));
	obj.push_back(o);
	img.push_back(a_tri);
	obj.push_back(o);
	img.push_back(b_tri);

	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector<cv::Mat> rvecs, tvecs;
	cv::calibrateCamera(obj, img, cv::Size(300, 450),
		cameraMatrix, distCoeffs, rvecs, tvecs, 0);
	std::cout << cameraMatrix << std::endl;
	std::cout << distCoeffs << std::endl;
	for (const cv::Mat &r : rvecs) std::cout << r * 180 / CV_PI << std::endl;
	for (const cv::Mat &t : tvecs) std::cout << t << std::endl;

	cv::Matx31f init_pnt(0, 0, 200);
	cv::Mat sphe = cv::Mat(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < rvecs.size(); ++i){
		cv::Mat r_mat;
		cv::Rodrigues(rvecs[i], r_mat);
		cv::Matx33f rotate_mat = r_mat;
		cv::Matx31f pnt = rotate_mat * init_pnt;
		cv::circle(sphe, cv::Point(round(pnt.val[0]) + 250, round(pnt.val[1]) + 250), 2, cv::Scalar(0, 0, 0), CV_FILLED);
		char count[100];
		sprintf(count, "%d", i);
		cv::putText(sphe, count, cv::Point(round(pnt.val[0]) + 250, round(pnt.val[1]) + 250), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	}
	cv::imshow("QQ", sphe);
	cv::imwrite(filename, sphe);
}

void test(){
	//FILE *graph_file = fopen("data/sampleFace0000-cycle-fix.curve", "r");
	//FILE *outfile = fopen("data/sampleFace0002-cycle-fix.curve", "w");
	//
	//unsigned int pnts_size;
	//fscanf(graph_file, "%u\n", &pnts_size);
	//fprintf(outfile, "%u\n", pnts_size);
	//for (unsigned int i = 0; i < pnts_size; ++i){
	//	double x, y;
	//	fscanf(graph_file, "V %lf %lf\n", &x, &y);
	//	fprintf(outfile, "V %lf %lf\n", -x, y);
	//}
	//
	//unsigned int edges_size;
	//fscanf(graph_file, "%u\n", &edges_size);
	//fprintf(outfile, "%u\n", edges_size);
	//for (unsigned int i = 0; i < edges_size; ++i){
	//	unsigned int p, q;
	//	fscanf(graph_file, "S %u %u\n", &p, &q);
	//	fprintf(outfile, "S %u %u\n", p, q);
	//}
	//fclose(graph_file);
	//fclose(outfile);
	//printf("=> Reading 0000 Writing 0002 done.\n");

	system("ansicon -E [30;43mtest ttt[0m");

	std::vector<int> a = std::vector<int>{2, 1, 0};
	std::vector<int> b = std::vector<int>{4, 5, 6};
	std::vector<int> c;
	std::reverse(a.begin(), a.end());

	c.push_back(3);
	c.insert(c.begin(), a.begin() + 1, a.end());
	c.insert(c.end(), b.begin(), b.end());
	c.resize(9);
	for (size_t i = 0; i < c.size(); ++i)
		printf("%d ", c[i]);
	printf("\n");

	std::unordered_set<int> us;
	us.insert(0);
	us.insert(1);
	us.insert(1);
	for (const int &num : us) printf("%d ", num);
	printf("size: %d\n", us.size());

	std::vector<int> d = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8};
	std::vector<int> e(d.begin(), d.begin() + 3);
	for (size_t i = 0; i < e.size(); ++i)
		printf("%d ", e[i]);
	printf("\n");

	//system("pause");
}

// https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
cv::Point3d rotate_with_axis(cv::Point3d v, double t, cv::Point3d p){
	double v_len = cv::norm(v);
	if (v_len < 0.0000001){
		printf("Error: v is too zero\n");
		return cv::Point3d(0, 0, 0);
	}
	v.x /= v_len, v.y /= v_len, v.z /= v_len;
	cv::Matx31d init_p(p);
	cv::Matx33d rotate_mat(
		cos(t) + (1 - cos(t)) * v.x * v.x, (1 - cos(t)) * v.x * v.y - sin(t) * v.z, (1 - cos(t)) * v.x * v.z + sin(t) * v.y,
		(1 - cos(t)) * v.y * v.x + sin(t) * v.z, cos(t) + (1 - cos(t)) * v.y * v.y, (1 - cos(t)) * v.y * v.z - sin(t) * v.x,
		(1 - cos(t)) * v.z * v.x - sin(t) * v.y, (1 - cos(t)) * v.z * v.y + sin(t) * v.x, cos(t) + (1 - cos(t)) * v.z * v.z
		);
	cv::Matx31d rotate_p = rotate_mat * init_p;
	return cv::Point3d(rotate_p.val[0], rotate_p.val[1], rotate_p.val[2]);
}

// triangle area
double herons_formula(cv::Point2d a, cv::Point2d b, cv::Point2d c){
	double ab = cv::norm(a - b);
	double bc = cv::norm(b - c);
	double ca = cv::norm(c - a);
	double s = (ab + bc + ca) / 2;

	return sqrt(s * (s - ab) * (s - bc) * (s - ca));
}

// return [0, PI]
double rotate_angle(double a_area, double b_area){
	printf("b/a: %lf\n", b_area / a_area);
	if (b_area / a_area > 1) return 0;
	return acos(b_area / a_area);
}

cv::Point2d predict_direct(cv::Mat &img, std::vector<cv::Point2f> &notable_pnts, char *filename){
	int ps = std::max(img.rows, img.cols);
	cv::Point2f ds(img.cols >> 1, img.rows >> 1);

	cv::Mat img_show, img_scale;
	img_show = img.clone();

	cv::circle(img_show, cv::Point2f(notable_pnts[0].x * ps, notable_pnts[0].y * -ps) + ds, 2,
		cv::Scalar(255, 0, 0), CV_FILLED);
	cv::circle(img_show, cv::Point2f(notable_pnts[5].x * ps, notable_pnts[5].y * -ps) + ds, 2,
		cv::Scalar(255, 0, 0), CV_FILLED);
	cv::circle(img_show, cv::Point2f(notable_pnts[6].x * ps, notable_pnts[6].y * -ps) + ds, 2,
		cv::Scalar(255, 0, 0), CV_FILLED);
	cv::line(img_show,
		cv::Point2f(notable_pnts[0].x * ps, notable_pnts[0].y * -ps) + ds,
		cv::Point2f(notable_pnts[5].x * ps, notable_pnts[5].y * -ps) + ds,
		cv::Scalar(200, 200, 0));
	cv::line(img_show,
		cv::Point2f(notable_pnts[5].x * ps, notable_pnts[5].y * -ps) + ds,
		cv::Point2f(notable_pnts[6].x * ps, notable_pnts[6].y * -ps) + ds,
		cv::Scalar(200, 200, 0));
	cv::line(img_show,
		cv::Point2f(notable_pnts[6].x * ps, notable_pnts[6].y * -ps) + ds,
		cv::Point2f(notable_pnts[0].x * ps, notable_pnts[0].y * -ps) + ds,
		cv::Scalar(200, 200, 0));

	cv::Point2f eyes_mid = notable_pnts[5] + notable_pnts[6];
	eyes_mid.x /= 2, eyes_mid.y /= 2;

	cv::Point2f face_mid = notable_pnts[0] + eyes_mid * 3;
	face_mid.x /= 4, face_mid.y /= 4;

	cv::Point2f width_vec = (notable_pnts[6] - eyes_mid) * 2;
	cv::Point2f height_vec = (notable_pnts[0] - face_mid) * 2;

	std::vector<cv::Point2f> rect_crn;
	rect_crn.resize(4);

	rect_crn[0] = face_mid - width_vec - height_vec;
	rect_crn[1] = face_mid + width_vec - height_vec;
	rect_crn[2] = face_mid + width_vec + height_vec;
	rect_crn[3] = face_mid - width_vec + height_vec;

	for (unsigned int i = 0; i < rect_crn.size(); ++i){
		cv::Point2f a_pnt = !i ? rect_crn.back() : rect_crn[i - 1];
		cv::Point2f b_pnt = rect_crn[i];
		cv::line(img_show,
			cv::Point2f(a_pnt.x * ps, a_pnt.y * -ps) + ds,
			cv::Point2f(b_pnt.x * ps, b_pnt.y * -ps) + ds,
			cv::Scalar(200, 0, 200));
	}
	cv::circle(img_show, cv::Point2f(face_mid.x * ps, face_mid.y * -ps) + ds, 2,
		cv::Scalar(0, 0, 255), CV_FILLED);
	cv::circle(img_show, cv::Point2f(notable_pnts[3].x * ps, notable_pnts[3].y * -ps) + ds, 2,
		cv::Scalar(0, 255, 0), CV_FILLED);

	cv::resize(img_show, img_scale, cv::Size(img_show.cols * 2, img_show.rows * 2));
	cv::imshow(filename, img_scale);
	cv::imwrite(filename, img_scale);

	return notable_pnts[3] - face_mid;
}

cv::Point2d get_vn(cv::Point2d vt){
	double vt_len = cv::norm(vt);
	vt.x /= vt_len, vt.y /= vt_len;
	cv::Point2d vn = cv::Point2d(vt.y, -vt.x);
	if (vt.cross(vn) < 0) vn *= -1;
	return vn;
}

void run_sphere(){
	int n;
	FILE *notable_file;

	cv::Mat sphe = cv::Mat(501, 501, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::line(sphe, cv::Point(0, 250), cv::Point(500, 250), cv::Scalar(0, 0, 150));
	cv::line(sphe, cv::Point(250, 0), cv::Point(250, 500), cv::Scalar(0, 0, 150));
	cv::Mat spher = cv::Mat(501, 501, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::line(spher, cv::Point(0, 250), cv::Point(500, 250), cv::Scalar(0, 0, 150));
	cv::line(spher, cv::Point(250, 0), cv::Point(250, 500), cv::Scalar(0, 0, 150));

	cv::Mat img;
	std::vector<cv::Point2f> notable_pnts;
	for (unsigned int i = 0; i < 7; ++i){
		char count[10], filename[100];
		sprintf(count, "%d", i);
		sprintf(filename, "data/%04d.png", i);
		img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		sprintf(filename, "data/notable_%04d.txt", i);
		notable_file = fopen(filename, "r");
		fscanf(notable_file, "%d", &n);
		notable_pnts.resize(n);
		for (unsigned int i = 0; i < n; ++i)
			fscanf(notable_file, "%f %f\n", &notable_pnts[i].x, &notable_pnts[i].y);
		fclose(notable_file);
		sprintf(filename, "pre_%04d.png", i);
		cv::Point2d vt = predict_direct(img, notable_pnts, filename);
		cv::circle(spher, cv::Point(round(vt.x * 3000) + 250, round(vt.y * -3000) + 250), 2, cv::Scalar(255, 100, 0), CV_FILLED);
		cv::putText(spher, count, cv::Point(round(vt.x * 3000) + 250, round(vt.y * -3000) + 250), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
		cv::Point2d vn = get_vn(vt);
		printf("vn: %lf, %lf\n", vt.x, vt.y);
		double t = rotate_angle(0.015931, herons_formula(notable_pnts[0], notable_pnts[5], notable_pnts[6]));
		printf("theta: %lf\n", t);
		cv::Point3d init_p(0, 0, 200);
		cv::Point3d r_p = rotate_with_axis(cv::Point3d(vn.x, vn.y, 0), t, init_p);
		if (i == 3) continue;
		cv::circle(sphe, cv::Point(round(r_p.x) + 250, round(r_p.y) * -1 + 250), 2, cv::Scalar(255, 100, 0), CV_FILLED);
		cv::putText(sphe, count, cv::Point(round(r_p.x) + 250, round(r_p.y) * -1 + 250), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	}
	cv::imshow("QQ", sphe);
	cv::imshow("QQQ", spher);
	cv::imwrite("sphere.png", sphe);
	cv::imwrite("sphere2.png", spher);

	cv::waitKey(0);
}

// mx + y = c
cv::Point2d get_ratio(std::vector<cv::Point2f> &notable_pnts){
	cv::Point2f n = notable_pnts[3] - notable_pnts[2];
	float m = -n.y / n.x;
	float c = m * notable_pnts[2].x + notable_pnts[2].y;
	cv::Point2f v = notable_pnts[1] - notable_pnts[0];
	float t = (c - m * notable_pnts[0].x - notable_pnts[0].y) / (m * v.x + v.y);
	cv::Point2f eyes_mid = notable_pnts[0] + v * t;
	cv::Point2d ratio;
	ratio.x = cv::norm(notable_pnts[3] - eyes_mid) / cv::norm(n); // width
	ratio.y = cv::norm(eyes_mid - notable_pnts[1]) / cv::norm(v * t); // height
	return ratio;
}

cv::Point2d hand_predict_direct(cv::Mat &img, std::vector<cv::Point2f> &notable_pnts, double width_ratio, double height_ratio, char *filename){
	int ps = std::max(img.rows, img.cols);
	cv::Point2f ds(img.cols >> 1, img.rows >> 1);

	cv::Mat img_show, img_scale;
	img_show = img.clone();

	cv::circle(img_show, notable_pnts[0] * ps + ds, 2,
		cv::Scalar(255, 0, 0), CV_FILLED);
	cv::circle(img_show, notable_pnts[2] * ps + ds, 2,
		cv::Scalar(255, 0, 0), CV_FILLED);
	cv::circle(img_show, notable_pnts[3] * ps + ds, 2,
		cv::Scalar(255, 0, 0), CV_FILLED);
	cv::line(img_show,
		notable_pnts[0] * ps + ds,
		notable_pnts[2] * ps + ds,
		cv::Scalar(200, 200, 0));
	cv::line(img_show,
		notable_pnts[2] * ps + ds,
		notable_pnts[3] * ps + ds,
		cv::Scalar(200, 200, 0));
	cv::line(img_show,
		notable_pnts[3] * ps + ds,
		notable_pnts[0] * ps + ds,
		cv::Scalar(200, 200, 0));

	cv::Point2f eyes_mid, face_mid;
	eyes_mid.x = notable_pnts[2].x * width_ratio + notable_pnts[3].x * (1 - width_ratio);
	eyes_mid.y = notable_pnts[2].y * width_ratio + notable_pnts[3].y * (1 - width_ratio);
	face_mid.x = notable_pnts[0].x * height_ratio + eyes_mid.x * (1 - height_ratio);
	face_mid.y = notable_pnts[0].y * height_ratio + eyes_mid.y * (1 - height_ratio);

	cv::Point2f width_vec = (notable_pnts[3] - eyes_mid) * 2;
	cv::Point2f height_vec = (notable_pnts[0] - face_mid) * 2;

	std::vector<cv::Point2f> rect_crn;
	rect_crn.resize(4);

	rect_crn[0] = face_mid - width_vec - height_vec;
	rect_crn[1] = face_mid + width_vec - height_vec;
	rect_crn[2] = face_mid + width_vec + height_vec;
	rect_crn[3] = face_mid - width_vec + height_vec;

	for (unsigned int i = 0; i < rect_crn.size(); ++i){
		cv::Point2f a_pnt = !i ? rect_crn.back() : rect_crn[i - 1];
		cv::Point2f b_pnt = rect_crn[i];
		cv::line(img_show,
			a_pnt * ps + ds,
			b_pnt * ps + ds,
			cv::Scalar(200, 0, 200));
	}
	cv::circle(img_show, face_mid * ps + ds, 2,
		cv::Scalar(0, 0, 255), CV_FILLED);
	cv::circle(img_show, notable_pnts[1] * ps + ds, 2,
		cv::Scalar(0, 255, 0), CV_FILLED);

	cv::resize(img_show, img_scale, cv::Size(img_show.cols * 2, img_show.rows * 2));
	cv::imshow(filename, img_scale);
	cv::imwrite(filename, img_scale);

	return notable_pnts[1] - face_mid;
}

void rotate_to_horizontal(){
	for (unsigned int i = 0; i < notable_pnts.size(); ++i){
		cv::Point2f v = notable_pnts[i][3] - notable_pnts[i][2];
		double theta = atan2(v.y, v.x);
		
		cv::Matx31f rvecs(0, 0, -theta);
		cv::Mat r_mat;
		cv::Rodrigues(rvecs, r_mat);
		cv::Matx33f rotate_mat = r_mat;
		for (unsigned int j = 0; j < notable_pnts[i].size(); ++j){
			cv::Matx31f init_pnt(notable_pnts[i][j].x, notable_pnts[i][j].y, 0);
			cv::Matx31f pnt = rotate_mat * init_pnt;
			notable_pnts[i][j].x = pnt.val[0];
			notable_pnts[i][j].y = pnt.val[1];
		}

		cv::Mat img_rotate_mat = cv::getRotationMatrix2D(cv::Point2d(imgs[i].cols >> 1, imgs[i].rows >> 1), theta / CV_PI * 180, 1);
		cv::warpAffine(imgs[i], imgs[i], img_rotate_mat, imgs[i].size());
	}

	return;
}

cv::Mat img, img_scale;
FILE *write_pnts;

void write_choose_pnt(int x, int y){
	int cx = img_scale.cols >> 1;
	int cy = img_scale.rows >> 1;
	int ps = std::max(img_scale.rows, img_scale.cols);

	fprintf(write_pnts, "%lf %lf\n", (x - cx) * 1.0 / ps, (y - cy) * 1.0 / ps);

	return;
}

void set_pnts_mouse_fun(int event, int x, int y, int flags, void *userdata){
	if (event == cv::EVENT_LBUTTONDOWN){
		write_choose_pnt(x, y);
	}

	return;
}

void show_pnts(){

	char show_name[10];
	for (unsigned int i = 0; i < notable_pnts.size(); ++i){
		cv::Point2f ds = cv::Point2f(imgs[i].cols >> 1, imgs[i].rows >> 1);
		int ps = std::max(imgs[i].rows, imgs[i].cols);
		cv::line(imgs[i], notable_pnts[i][0] * ps + ds, notable_pnts[i][2] * ps + ds, cv::Scalar(255, 255, 0));
		cv::line(imgs[i], notable_pnts[i][2] * ps + ds, notable_pnts[i][3] * ps + ds, cv::Scalar(255, 255, 0));
		cv::line(imgs[i], notable_pnts[i][3] * ps + ds, notable_pnts[i][0] * ps + ds, cv::Scalar(255, 255, 0));
		cv::circle(imgs[i], notable_pnts[i][1] * ps + ds, 2, cv::Scalar(0, 255, 0), CV_FILLED);

		cv::resize(imgs[i], img_scale, cv::Size(imgs[i].cols * 3, imgs[i].rows * 3));
		sprintf(show_name, "show_%02d", i);
		cv::imshow(show_name, img_scale);
	}
	cv::waitKey(0);
}

/*
*   Lw = Dw - Aw
*	Dw = sum(w(i,.))
*	Aw = w(i,j)
*/
void ringIt(){
	unsigned int n = notable_pnts.size(), m = notable_pnts[0].size();
	std::vector<std::vector<double>> Lw;
	Lw.resize(n);
	for (unsigned int i = 0; i < n; ++i)
		Lw[i].resize(n);
	// - Aw
	double t = 0;
	for (int i = 0; i < n; ++i){
		for (int j = i + 1; j < n; ++j){
			double kp_diff = 0;
			for (int k = 0; k < m; ++k){
				Lw[i][j] = Lw[j][i] += cv::norm(notable_pnts[i][k] - notable_pnts[j][k]);
				kp_diff += Lw[i][j];
			}
			t += kp_diff;
		}
	}
	t /= n * (n - 1) / 2;

	for (int i = 0; i < n; ++i){
		for (int j = i; j < n; ++j){
			Lw[i][j] = Lw[j][i] = -exp(-Lw[i][j] / t);
			//std::cout << Lw[i][j] << "\t";
		}
		//std::cout << std::endl;
	}

	// Lw = - Aw - (-Dw)
	for (int i = 0; i < n; ++i){
		double sumD = 0;
		// -Dw
		for (int j = 0; j < n; ++j){
			sumD += Lw[i][j];
		}
		Lw[i][i] -= sumD;
	}

	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j)
			std::cout << Lw[i][j] << "\t";
		std::cout << std::endl;
	}

	cv::Mat e = cv::Mat(n, n, CV_32FC1);
	for (int i = 0; i < n; ++i)
	for (int j = 0; j < n; ++j)
		((float *)e.data)[i * e.cols + j] = (float)Lw[i][j];
	std::cout << e << std::endl;

	cv::Mat eVal, eVec;
	eigen(e, eVal, eVec);
	std::cout << eVec << std::endl;

	cv::Mat ring = cv::Mat(1000, 1000, CV_32FC1, cv::Scalar(1));
	int c = 500, r = 500;
	circle(ring, cv::Point(c, c), 1, 0, 2);

	for (int i = 0; i < n; ++i){
		circle(ring, cv::Point(c + r * ((float *)eVec.data)[(n - 2) * e.cols + i], c + r * ((float *)eVec.data)[(n - 3) * e.cols + i]), 1, 0, 2);
		char text[10];
		itoa(i, text, 10);
		putText(ring, cv::String(text), cv::Point(c + r*((float *)eVec.data)[(n - 2) * e.cols + i], c + r*((float *)eVec.data)[(n - 3) * e.cols + i]), cv::FONT_HERSHEY_DUPLEX, 1, 0);

		// use atan2f turn to angle, use (270 - a) % 360 retarget x-axie
		//img_angle.push_back(im_ang(i, fmod(270 - atan2f(((float *)eVec.data)[(img_seq.size() - 3) * e.cols + i], ((float *)eVec.data)[(img_seq.size() - 2) * e.cols + i]) * 180 / M_PI, 360)));
		//std::cout << fmod(270 - atan2f(((float *)eVec.data)[(img_seq.size() - 3) * e.cols + i], ((float *)eVec.data)[(img_seq.size() - 2) * e.cols + i]) * 180 / M_PI, 360) << std::endl;
	}

	//std::sort(img_angle.begin(), img_angle.end(), im_ang_cmp);
	//for (int i = 0; i < img_angle.size(); ++i)
	//	std::cout << img_angle[i].index << ", " << img_angle[i].angle << std::endl;
	cv::imshow("RingIt", ring);
	cv::Mat ring8U;
	ring.convertTo(ring8U, CV_8U, 255.0 / 1.0);
	cv::imwrite("RingIt.png", ring8U);

	cv::waitKey(0);
}

void run_hand_sphere(char *data_name){

	cv::Mat sphe = cv::Mat(501, 501, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::line(sphe, cv::Point(0, 250), cv::Point(500, 250), cv::Scalar(0, 0, 150));
	cv::line(sphe, cv::Point(250, 0), cv::Point(250, 500), cv::Scalar(0, 0, 150));
	cv::Mat spher = cv::Mat(501, 501, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::line(spher, cv::Point(0, 250), cv::Point(500, 250), cv::Scalar(0, 0, 150));
	cv::line(spher, cv::Point(250, 0), cv::Point(250, 500), cv::Scalar(0, 0, 150));

	cv::Point2d standard_ratio;
	double standard_area = 0;
	char count[10], filename[100];
	for (unsigned int i = 0; i < notable_pnts.size(); ++i){
		sprintf(count, "%d", i);
		sprintf(filename, "data/%s/pre_%02d.png", data_name, i);
		if (!i) standard_ratio = get_ratio(notable_pnts[i]);
		cv::Point2d vt = hand_predict_direct(imgs[i], notable_pnts[i], standard_ratio.x, standard_ratio.y, filename);
		cv::circle(spher, cv::Point(round(vt.x * 3000) + 250, round(vt.y * 3000) + 250), 2, cv::Scalar(255, 100, 0), CV_FILLED);
		cv::putText(spher, count, cv::Point(round(vt.x * 3000) + 250, round(vt.y * 3000) + 250), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
		cv::Point2d vn = get_vn(vt);
		if (!i) standard_area = herons_formula(notable_pnts[i][0], notable_pnts[i][2], notable_pnts[i][3]);
		double t = rotate_angle(standard_area, herons_formula(notable_pnts[i][0], notable_pnts[i][2], notable_pnts[i][3]));
		cv::Point3d init_p(0, 0, 200);
		cv::Point3d r_p = rotate_with_axis(cv::Point3d(vn.x, vn.y, 0), t, init_p);
		cv::circle(sphe, cv::Point(round(r_p.x) + 250, round(r_p.y) + 250), 2, cv::Scalar(255, 100, 0), CV_FILLED);
		cv::putText(sphe, count, cv::Point(round(r_p.x) + 250, round(r_p.y) + 250), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	}
	cv::imshow("sphere", sphe);
	//cv::imshow("QQQ", spher);
	sprintf(filename, "data/%s/sphere.png", data_name);
	cv::imwrite(filename, sphe);
	//cv::imwrite("sphere2.png", spher);

	cv::waitKey(0);
}

void read_notable(unsigned int n, unsigned int n_size, char *data_name){
	FILE *notable_file;
	char filename[100];
	notable_pnts.resize(n), imgs.resize(n);
	for (unsigned int i = 0; i < n; ++i){
		sprintf(filename, "data/%s/%02d.jpg", data_name, i);
		imgs[i] = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		sprintf(filename, "data/%s/%02d.txt", data_name, i);
		notable_file = fopen(filename, "r");
		notable_pnts[i].resize(n_size);
		for (unsigned int j = 0; j < n_size; ++j)
			fscanf(notable_file, "%f %f\n", &notable_pnts[i][j].x, &notable_pnts[i][j].y);
		fclose(notable_file);
	}
	return;
}

void run_set_pnts(char *data_name, unsigned int n){
	char filename[100];
	sprintf(filename, "data/%s/%02d.jpg", data_name, n);
	img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::resize(img, img_scale, cv::Size(img.cols * 2, img.rows * 2));

	sprintf(filename, "data/%s/%02d.txt", data_name, n);
	write_pnts = fopen(filename, "w");

	cv::namedWindow("set pnts", cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback("set pnts", set_pnts_mouse_fun, NULL);
	cv::imshow("set pnts", img_scale);
	cv::waitKey(0);

	fclose(write_pnts);
}

cv::Mat find_path_img;
int SET_FRAME_FLAG = -2;
std::vector<cv::Point2d> graph_pnts;
std::vector<int> frame_index;
std::vector<std::vector<double>> complete_graph;
std::vector<std::vector<bool>> mst_graph;
void set_graph_pnt(int x, int y){
	graph_pnts.push_back(cv::Point2d(x, y));
	cv::circle(find_path_img, cv::Point(x, y), 2, cv::Scalar(0, 50, 200), CV_FILLED);
	cv::imshow("find path", find_path_img);

	return;
}

void build_complete_graph(){
	complete_graph = std::vector<std::vector<double>>();
	complete_graph.resize(graph_pnts.size());
	for (auto &cg : complete_graph) cg.resize(graph_pnts.size());

	for (unsigned int i = 0; i < graph_pnts.size(); ++i)
		for (unsigned int j = i + 1; j < graph_pnts.size(); ++j)
			complete_graph[i][j] = complete_graph[j][i] = cv::norm(graph_pnts[i] - graph_pnts[j]);

	for (unsigned int i = 0; i < graph_pnts.size(); ++i)
		for (unsigned int j = i + 1; j < graph_pnts.size(); ++j)
			printf("%d, %d = %lf\n", i, j, complete_graph[i][j]);

	system("ansicon -e [42mInfo:[0m Build Complete Graph");
	return;
}


void optimize_mst(){
	unsigned int n = complete_graph.size();

	IloEnv env;

	IloModel model(env);
	IloArray<IloBoolVarArray> x(env, n);
	for (unsigned int i = 0; i < n; ++i){
		x[i] = IloBoolVarArray(env, n);
		for (unsigned int j = 0; j < n; ++j)
			x[i][j] = IloBoolVar(env);
	}

	IloExpr expr(env);
	for (unsigned int i = 0; i < n; ++i)
		for (unsigned int j = 0; j < n; ++j)
			expr += x[i][j] * complete_graph[i][j];

	model.add(IloMinimize(env, expr));

	IloArray<IloArray<IloBoolVarArray>> y(env, n);
	for (unsigned int i = 0; i < n; ++i){
		y[i] = IloArray<IloBoolVarArray>(env, n);
		for (unsigned int j = 0; j < n; ++j){
			y[i][j] = IloBoolVarArray(env, n);
			for (unsigned int k = 0; k < n; ++k)
				y[i][j][k] = IloBoolVar(env);
		}
	}
	
	IloConstraintArray c(env);
	// not using constrain
	for (unsigned int i = 0; i < n; ++i)
		c.add(x[i][i] == 0);
	for (unsigned int i = 0; i < n; ++i)
		for (unsigned int j = i + 1; j < n; ++j)
			c.add(x[j][i] == x[i][j]);
	// edge constrain
	IloBoolVarArray sum_x(env);
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = i + 1; j < n; ++j){
			sum_x.add(x[i][j]);
		}
	}
	c.add(IloSum(sum_x) == n - 1);
	// m2 constrain
	for (unsigned int i = 0; i < n; ++i)
	for (unsigned int j = i + 1; j < n; ++j)
	for (unsigned int k = 0; k < n; ++k)
		c.add(y[i][j][k] + y[j][i][k] == x[i][j]);
	// m3 constrain
	for (unsigned int i = 0; i < n; ++i)
	for (unsigned int j = i + 1; j < n; ++j){
		IloBoolVarArray sum_y(env);
		for (unsigned int k = 0; k < n; ++k)
			sum_y.add(y[i][k][j]);
		c.add(IloSum(sum_y) + x[i][j] == 1);
	}

	model.add(c);


	IloCplex cplex(model);
	//cplex.setOut(env.getNullStream());
	if (!cplex.solve()) {
		env.error() << "Failed to optimize LP" << std::endl;
	}

	IloNumArray vals(env);
	for (unsigned int i = 0; i < n; ++i){
		cplex.getValues(vals, x[i]);
		env.out() << "[" << i << "] = " << vals << std::endl;
		for (unsigned int j = 0; j < n; ++j)
		if (vals[j] == 1){
			cv::line(find_path_img, graph_pnts[i], graph_pnts[j], cv::Scalar(50, 200, 50), 2);
			mst_graph[i][j] = mst_graph[i][j] = true;
		}
	}

	//env.out() << "Solution status = " << cplex.getStatus() << std::endl;
	//env.out() << "Solution value  = " << cplex.getObjValue() << std::endl;
	//cplex.getValues(vals, x);
	//env.out() << "Values        = " << vals << std::endl;
	//cplex.getSlacks(vals, c);
	//env.out() << "Slacks        = " << vals << std::endl;
	//cplex.getDuals(vals, c);
	//env.out() << "Duals         = " << vals << std::endl;
	//cplex.getReducedCosts(vals, x);
	//env.out() << "Reduced Costs = " << vals << std::endl;

	env.end();

	cv::imshow("find path", find_path_img);

	return;
}

// Kruskal's algorithm
// http://www.csie.ntnu.edu.tw/~u91029/SpanningTree.html
std::vector<unsigned int> root_set;
unsigned int find_root(unsigned int n){ return n == root_set[n] ? n : find_root(root_set[n]); };
void union_root(unsigned int n, unsigned int m){ root_set[find_root(n)] = find_root(m); };
void build_minimum_spanning_tree(){ // need degree constrain
	mst_graph = std::vector<std::vector<bool>>();
	mst_graph.resize(graph_pnts.size());
	for (auto &mg : mst_graph) mg.resize(graph_pnts.size());

	typedef struct edge{
		unsigned int i, j;
		double length;
		edge(unsigned int ti, unsigned int tj, double tl){ i = ti, j = tj, length = tl; };
	}edge;
	struct edge_small_cmp{ bool operator()(edge const &a, edge const &b){ return a.length < b.length; }; };
	
	std::vector<edge> edges; // i ¥Ã»·¤p©ó j
	for (unsigned int i = 0; i < graph_pnts.size(); ++i)
		for (unsigned int j = i + 1; j < graph_pnts.size(); ++j)
			edges.push_back(edge(i, j, complete_graph[i][j]));
	std::sort(edges.begin(), edges.end(), edge_small_cmp());

	root_set.resize(graph_pnts.size());
	for (unsigned int i = 0; i < root_set.size(); ++i) root_set[i] = i;

	for (unsigned int i = 0, j = 0; i < graph_pnts.size() - 1; ++i){
		while (find_root(edges[j].i) == find_root(edges[j].j)
			// || edges[j].i == 0 && edges[j].j == graph_pnts.size() - 1 // ¥i¥H¥[¤J­þ¨ÇÂI¤£¬Û¾Fªº­­¨î
			) ++j; 
		union_root(edges[j].i, edges[j].j);
		mst_graph[edges[j].i][edges[j].i] = mst_graph[edges[j].i][edges[j].i] = true;
		cv::line(find_path_img, graph_pnts[edges[j].i], graph_pnts[edges[j].j], cv::Scalar(50, 200, 50), 2);
		++j;
	}

	cv::imshow("find path", find_path_img);
	system("ansicon -e [42mInfo:[0m Build Minimum Spanning Tree.");
	return;
}

void set_pnt_frame(int frame){

}

void find_path_mouse_fun(int event, int x, int y, int flags, void *userdata){
	if (event == cv::EVENT_LBUTTONDOWN && SET_FRAME_FLAG == -2){
		set_graph_pnt(x, y);
	}
	else if (event == cv::EVENT_LBUTTONDOWN && SET_FRAME_FLAG > -2){
		set_pnt_frame(SET_FRAME_FLAG);
	}

	return;
}

void find_path(){
	SET_FRAME_FLAG = -2;
	graph_pnts = std::vector<cv::Point2d>();
	find_path_img = cv::Mat(501, 501, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::line(find_path_img, cv::Point(0, 250), cv::Point(500, 250), cv::Scalar(0, 0, 150));
	cv::line(find_path_img, cv::Point(250, 0), cv::Point(250, 500), cv::Scalar(0, 0, 150));
	cv::namedWindow("find path", cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback("find path", find_path_mouse_fun, NULL);
	cv::imshow("find path", find_path_img);
	while (1){
		char key;
		key = cv::waitKey(0);
		if (key == 27) break;
		if (key == 'a') SET_FRAME_FLAG = -2;
		if (key == 'b') SET_FRAME_FLAG = 0;
		if (key == 'e') SET_FRAME_FLAG = -1;
		if (key == 'c') build_complete_graph();
		if (key == 't') build_minimum_spanning_tree();
		if (key == 'o') optimize_mst();
	}

	return;
}

int main(){
	find_path();
	//run_set_pnts("07_women", 0);
	//read_notable(15, 4, "07_women");
	//rotate_to_horizontal();
	//show_pnts();
	//run_hand_sphere("07_women");
	//ringIt();

	return 0;
}