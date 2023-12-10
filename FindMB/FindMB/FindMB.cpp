#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include "Vector2Int.h"

using namespace std;
using namespace cv;

Mat CutImage(Mat image, Vector2Int pos, Vector2Int size);
int CaculateSumDifferant(Mat f1, Mat f2);
Mat CaculateMV(Mat f1, Mat f2);
void MergeImage(Mat image, Mat target, Vector2Int pos);
void DrawVector(Mat image, Vector2Int direction);
void ShowMV(vector<Mat> frames);


int main() {
	VideoCapture cap("dance.mp4");

	if (!cap.isOpened()) {
		cerr << "cannot open video" << endl;
		return -1;
	}

	vector<Mat> frames;

	int i = 2;
	while (i > 0)
	{
		i--;
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			break;
		}
		frames.push_back(frame);
	}

	imshow("Frame 1", frames[0]);
	imshow("Frame 2", frames[1]);
	imshow("MV", CaculateMV(frames[0], frames[1]));


	waitKey(0);

	cap.release();
	destroyAllWindows();

	return 0;
}

void ShowMV(vector<Mat> frames) {
	vector<Mat> mvs;
	for (int i = 0; i < frames.size() - 1; i++)
	{
		auto mv = CaculateMV(frames[i], frames[i + 1]);
		mvs.push_back(mv);

		imshow("Mv" + i, mv);
	}
}

Mat CaculateMV(Mat f1, Mat f2)
{
	int rows = f1.rows;
	int cols = f1.cols;

	Mat res = Mat(rows, cols, f1.type(), Scalar::all(255));
	int mbLenght = 16;
	int searchLength = 8;

	for (int i = 0; i < rows; i += mbLenght)
	{
		for (int j = 0; j < cols; j += mbLenght)
		{
			auto mbf1 = CutImage(f1, Vector2Int(i, j), Vector2Int(mbLenght, mbLenght));

			int minDif = INT_MAX;
			Mat minDifMb;
			Vector2Int minDifPos;

			for (int x = i - searchLength; x < i + mbLenght; x++)
			{
				for (int y = j - searchLength; y < j + mbLenght; y++)
				{
					auto checkMb = CutImage(f2, Vector2Int(x, y), Vector2Int(mbLenght, mbLenght));
					
					int sumDiff = CaculateSumDifferant(mbf1, checkMb);
					if (sumDiff < minDif)
					{
						minDif = sumDiff;
						minDifMb = checkMb;
						minDifPos = Vector2Int(x, y);
					}
				}
			}

			Vector2Int mv = minDifPos - Vector2Int(i, j);
			//Mat f2p = CutImage(f1, minDifPos, Vector2Int(mbLenght, mbLenght)) - mbf1;
			//Mat deltaF2 = CutImage(f2, Vector2Int(i, j), Vector2Int(mbLenght, mbLenght)) - f2p;

			Mat mvBlock = Mat(mbLenght, mbLenght, f1.type(), Scalar::all(255));
			DrawVector(mvBlock, mv);
			MergeImage(res, mvBlock, Vector2Int(i, j));
		}
	}
	return res;
}

Mat CutImage(Mat image, Vector2Int pos, Vector2Int size)
{
	Mat res = Mat(size.x, size.y, image.type());
	for (int i = pos.x, x = 0; i < pos.x + size.x; i++, x++)
	{
		for (int j = pos.y, y = 0; j < pos.y + size.y; j++, y++)
		{
			if (i >= image.rows || j >= image.cols || i < 0 || j < 0)
			{
				res.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
			}
			else
			{
				res.at<Vec3b>(x, y) = image.at<Vec3b>(i, j);
			}
		}
	}
	return res;
}

void MergeImage(Mat image, Mat target, Vector2Int pos)
{
	for (int i = pos.x, x = 0; i < pos.x + target.rows; i++, x++)
	{
		for (int j = pos.y, y = 0; j < pos.y + target.cols; j++, y++)
		{
			if (i >= image.rows || j >= image.cols || i < 0 || j < 0)
				continue;
			image.at<Vec3b>(i, j) = target.at<Vec3b>(x, y);
		}
	}
}

int CaculateSumDifferant(Mat f1, Mat f2)
{
	int rows = f1.rows;
	int cols = f1.cols;
	int res = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			auto p1 = f1.at<Vec3b>(i, j);
			auto p2 = f2.at<Vec3b>(i, j);

			res += abs((int)p1[2] - (int)p2[2]) + abs((int)p1[1] - (int)p2[1]) + abs((int)p1[0] - (int)p2[0]);
		}
	}
	return res;
}

void DrawVector(Mat image, Vector2Int direction)
{
	if (image.empty()) {
		std::cerr << "Invalid image." << std::endl;
		return;
	}
	int width = image.cols;
	int height = image.rows;

	cv::Point center(width / 2, height / 2);

	int length = std::min(width, height) / 2;
	cv::Point endPoint1(center.x + direction.x * length, center.y - direction.y * length);
	//cv::Point endPoint2(center.x - direction.x * length, center.y + direction.y * length);

	cv::line(image, center, endPoint1, cv::Scalar(0, 0, 0), 1);
	//cv::line(image, center, endPoint2,cv::Scalar(0, 0, 0), 1);
}