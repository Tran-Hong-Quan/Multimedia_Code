#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

cv::Mat AverageFillter(cv::Mat image);
cv::Mat MedianFillter(cv::Mat image);
cv::Mat SoftMedianFillter(cv::Mat image);
cv::Mat LaplacianFillter(cv::Mat image);
cv::Mat Grayscale(cv::Mat image);
cv::Mat SobelFillter(cv::Mat image);
cv::Vec3b Brightness2Color(uchar brightness);
cv::Mat Histogram(cv::Mat image);
cv::Mat HistogramColor(cv::Mat image);

std::vector<int> Array2Vector(int array[],int len);
void PrintColor(cv::Vec3b pixel);
void Plot(std::vector<int> data, std::string name);


int main(int argc, char** argv)
{
	cv::Mat image = cv::imread("Noise/1.jpg");

	if (image.empty()) {
		std::cerr << "Cannot read image" << std::endl;
		return -1;
	}

	cv::Mat grayscaleImage;
	cv::Mat bluredImage;
	cv::Mat medianImage;
	cv::Mat laplaceImage;
	cv::Mat sobelImage;
	cv::Mat sobelGradient;
	cv::Mat histogramImage;
	cv::Mat unsharpImg;

	//grayscaleImage = Grayscale(image);
	//bluredImage = AverageFillter(image);
	medianImage = MedianFillter(image);
	//medianImage = SoftMedianFillter(image);
	//laplaceImage = LaplacianFillter(image);
	//sobelImage = SobelFillter(image);
	//histogramImage = Histogram(image);
	//histogramImage = HistogramColor(image);
	//unsharpImg = image + (image - bluredImage) * 21;

	cv::imshow("Original Image", image);
	//cv::imshow("Grayscale Image", grayscaleImage);
	//cv::imshow("Blur Image", bluredImage);
	cv::imshow("Median Image", medianImage);
	//cv::imshow("Laplace Image", laplaceImage);
	//cv::imshow("Sobel Image", sobelImage);
	//cv::imshow("Sobel Gradient", sobelGradient);
	//cv::imshow("Histogram Image", histogramImage);
	//cv::imshow("Unsharp Image", unsharpImg);

	cv::waitKey(0);

	return 0;
}

cv::Mat AverageFillter(cv::Mat image) {
	int rows = image.rows;
	int cols = image.cols;
	auto res = cv::Mat(rows, cols, image.type());

	int mask[3][3] = {
		1,2,1,
		2,4,2,
		1,2,1,
	};
	int maskRows = 3;
	int maskCols = 3;

	float demultiplexer = 1.0 / 16;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int sum[3] = { 0,0,0 };
			for (int x = 0; x < maskRows; x++)
			{
				for (int y = 0; y < maskCols; y++)
				{
					int imageX = i - (int)maskRows / 2 + x;
					int imageY = j - (int)maskCols / 2 + y;

					if (imageX < 0 || imageY < 0 || imageX >= rows || imageY >= cols)
						continue;

					auto pixel = image.at<cv::Vec3b>(imageX, imageY);

					sum[2] += (int)pixel[2] * mask[x][y];
					sum[1] += (int)pixel[1] * mask[x][y];
					sum[0] += (int)pixel[0] * mask[x][y];
				}
			}
			sum[2] *= demultiplexer; sum[1] *= demultiplexer; sum[0] *= demultiplexer;
			if (sum[2] > 255) sum[2] = 255; if (sum[1] > 255) sum[1] = 255; if (sum[0] > 255) sum[0] = 255;
			auto newPixel = cv::Vec3b((uchar)sum[0], (uchar)sum[1], (uchar)sum[2]);
			res.at<cv::Vec3b>(i, j) = newPixel;
		}
	}
	return res;
}

cv::Mat MedianFillter(cv::Mat image) {
	int rows = image.rows;
	int cols = image.cols;
	auto res = cv::Mat(rows, cols, image.type());

	int maskRows = 3;
	int maskCols = 3;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			std::vector<uchar> rList;
			std::vector<uchar> gList;
			std::vector<uchar> bList;

			for (int x = 0; x < maskRows; x++)
			{
				for (int y = 0; y < maskCols; y++)
				{
					int imageX = i - (int)maskRows / 2 + x;
					int imageY = j - (int)maskCols / 2 + y;

					if (imageX < 0 || imageY < 0 || imageX >= rows || imageY >= cols)
					{
						rList.push_back(0);
						gList.push_back(0);
						bList.push_back(0);
						continue;
					}
					auto pixel = image.at<cv::Vec3b>(imageX, imageY);
					rList.push_back(pixel[2]);
					gList.push_back(pixel[1]);
					bList.push_back(pixel[0]);

				}
			}

			std::sort(rList.begin(), rList.end());
			std::sort(gList.begin(), gList.end());
			std::sort(bList.begin(), bList.end());

			res.at<cv::Vec3b>(i, j) = cv::Vec3b(bList[bList.size() / 2], gList[gList.size() / 2], rList[rList.size() / 2]);
		}
	}
	return res;
}

cv::Mat SoftMedianFillter(cv::Mat image) {
	int rows = image.rows;
	int cols = image.cols;
	auto res = image.clone();

	int maskRows = 3;
	int maskCols = 3;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			std::vector<uchar> rList;
			std::vector<uchar> gList;
			std::vector<uchar> bList;

			for (int x = 0; x < maskRows; x++)
			{
				for (int y = 0; y < maskCols; y++)
				{
					int imageX = i - (int)maskRows / 2 + x;
					int imageY = j - (int)maskCols / 2 + y;

					if (imageX < 0 || imageY < 0 || imageX >= rows || imageY >= cols)
					{
						rList.push_back(0);
						gList.push_back(0);
						bList.push_back(0);
						continue;
					}

					auto pixel = res.at<cv::Vec3b>(imageX, imageY);
					rList.push_back(pixel[2]);
					gList.push_back(pixel[1]);
					bList.push_back(pixel[0]);

				}
			}

			std::sort(rList.begin(), rList.end());
			std::sort(gList.begin(), gList.end());
			std::sort(bList.begin(), bList.end());

			res.at<cv::Vec3b>(i, j) = cv::Vec3b(bList[bList.size() / 2], gList[gList.size() / 2], rList[rList.size() / 2]);
		}
	}
	return res;
}

cv::Mat Grayscale(cv::Mat image) {
	auto res = image.clone();
	int rows = res.rows;
	int cols = res.cols;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			auto pixel = res.at<cv::Vec3b>(i, j);
			uchar gray = (uchar)((double)pixel[2] * 0.299 + (double)pixel[1] * 0.587 + (double)pixel[0] * 0.114);
			res.at<cv::Vec3b>(i, j) = cv::Vec3b(gray, gray, gray);
		}
	}
	return res;
}

cv::Mat LaplacianFillter(cv::Mat image) {
	int rows = image.rows;
	int cols = image.cols;
	auto ref = Grayscale(image);
	auto res = cv::Mat(rows, cols, image.type());

	int mask[3][3] = {
		  0, -1,  0,
		 -1,  4, -1,
		  0, -1,  0,
	};
	int maskRows = 3;
	int maskCols = 3;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int sum = 0;
			for (int x = 0; x < maskRows; x++)
			{
				for (int y = 0; y < maskCols; y++)
				{
					int imageX = i - (int)maskRows / 2 + x;
					int imageY = j - (int)maskCols / 2 + y;

					if (imageX < 0 || imageY < 0 || imageX >= rows || imageY >= cols)
						continue;
					sum += ((int)ref.at<cv::Vec3b>(imageX, imageY)[0]) * mask[x][y];
				}
			}
			if (sum < 0) sum *= -1; if (sum > 255) sum = 255;
			auto resSum = (uchar)sum;
			res.at<cv::Vec3b>(i, j) = cv::Vec3b(resSum, resSum, resSum);
		}
	}
	return res;
}

cv::Mat SobelFillter(cv::Mat image)
{
	int rows = image.rows;
	int cols = image.cols;
	auto ref = Grayscale(image);
	auto Gx = cv::Mat(rows, cols, image.type());
	auto Gy = cv::Mat(rows, cols, image.type());
	auto res = cv::Mat(rows, cols, image.type());
	auto gradient = cv::Mat(rows, cols, image.type());

	int maskX[3][3] = {
		  1,  0, -1,
		  2,  0, -2,
		  1,  0, -1,
	};
	int maskY[3][3] = {
		  1,  2,  1,
		  0,  0,  0,
		 -1, -2, -1,
	};
	int maskRows = 3;
	int maskCols = 3;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int sumX = 0;
			int sumY = 0;
			for (int x = 0; x < maskRows; x++)
			{
				for (int y = 0; y < maskCols; y++)
				{
					int imageX = i - (int)maskRows / 2 + x;
					int imageY = j - (int)maskCols / 2 + y;

					if (imageX < 0 || imageY < 0 || imageX >= rows || imageY >= cols)
						continue;
					sumX += ((int)ref.at<cv::Vec3b>(imageX, imageY)[0]) * maskX[x][y];
					sumY += ((int)ref.at<cv::Vec3b>(imageX, imageY)[0]) * maskY[x][y];
				}
			}
			auto resSum = ((int)sqrt(sumX * sumX + sumY * sumY));

			if (resSum > 255) resSum = 255;

			res.at<cv::Vec3b>(i, j) = cv::Vec3b((uchar)resSum, (uchar)resSum, (uchar)resSum);
			gradient.at<cv::Vec3b>(i, j) = Brightness2Color((uchar)resSum);
		}
	}
	return res;
}

cv::Mat Histogram(cv::Mat image) {
	auto ref = image.clone();
	ref = Grayscale(ref);
	auto rows = ref.rows;
	auto cols = ref.cols;
	auto res = cv::Mat(rows, cols, ref.type());

	int p[256] = { 0 };
	for (int i = 0; i < ref.rows; i++) {
		for (int j = 0; j < ref.cols; j++)
		{
			p[(int)ref.at<cv::Vec3b>(i, j)[0]]++;
		}
	}
	//Plot(Array2Vector(p,256),"P");
	int cdf[256] = { 0 };
	int cdfMin = 0;
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			cdf[i] += p[j];
		}
		if (cdfMin == 0 && cdf[i] != 0)
		{
			cdfMin = cdf[i];
		}
		else if (cdfMin > cdf[i])
		{
			cdfMin = cdf[i];
		}
	}
	//Plot(Array2Vector(cdf, 256), "cdf");
	int h[256];
	for (int i = 0; i < 256; i++)
	{
		h[i] = (float)(cdf[i] - cdfMin) / (rows * cols - cdfMin) * 255;
	}
	for (int i = 0; i < ref.rows; i++) {
		for (int j = 0; j < ref.cols; j++)
		{
			auto hIndex = ((int)ref.at<cv::Vec3b>(i, j)[0]);
			auto hValue = h[hIndex];
			res.at<cv::Vec3b>(i, j) = cv::Vec3b((uchar)hValue, (uchar)hValue, (uchar)hValue);
		}
	}
	//int pRes[256] = { 0 };
	//for (int i = 0; i < ref.rows; i++) {
	//	for (int j = 0; j < ref.cols; j++)
	//	{
	//		pRes[(int)res.at<cv::Vec3b>(i, j)[0]]++;
	//	}
	//}
	//Plot(Array2Vector(pRes, 256), "P Res");
	return res;
}

cv::Mat HistogramColor(cv::Mat image) {
	auto ref = image.clone();
	auto rows = ref.rows;
	auto cols = ref.cols;
	auto res = cv::Mat(rows, cols, ref.type());

	int p[3][256] = { 0 };
	for (int i = 0; i < ref.rows; i++) {
		for (int j = 0; j < ref.cols; j++)
		{
			auto pixel = ref.at<cv::Vec3b>(i, j);
			p[2][(int)pixel[2]]++;
			p[1][(int)pixel[1]]++;
			p[0][(int)pixel[0]]++;
		}
	}
	int cdf[3][256] = { 0 };
	int cdfMin[3] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			cdf[2][i] += p[2][j];
			cdf[1][i] += p[1][j];
			cdf[0][i] += p[0][j];
		}
		if (cdfMin[2] == 0 && cdf[2][i] != 0) cdfMin[2] = cdf[2][i]; 
		else if (cdfMin[2] > cdf[2][i]) cdfMin[2] = cdf[2][i];

		if (cdfMin[1] == 0 && cdf[1][i] != 0) cdfMin[1] = cdf[1][i];
		else if (cdfMin[1] > cdf[1][i]) cdfMin[1] = cdf[1][i];

		if (cdfMin[0] == 0 && cdf[0][i] != 0) cdfMin[0] = cdf[0][i];
		else if (cdfMin[0] > cdf[0][i]) cdfMin[0] = cdf[0][i];
	}
	int h[3][256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		h[2][i] = (float)(cdf[2][i] - cdfMin[2]) / (rows * cols - cdfMin[2]) * 255;
		h[1][i] = (float)(cdf[1][i] - cdfMin[1]) / (rows * cols - cdfMin[1]) * 255;
		h[0][i] = (float)(cdf[0][i] - cdfMin[0]) / (rows * cols - cdfMin[0]) * 255;
	}
	for (int i = 0; i < ref.rows; i++) {
		for (int j = 0; j < ref.cols; j++)
		{
			auto hIndex = ref.at<cv::Vec3b>(i, j);
			auto hValueR = h[2][(int)hIndex[2]];
			auto hValueG = h[1][(int)hIndex[1]];
			auto hValueB = h[0][(int)hIndex[0]];
			res.at<cv::Vec3b>(i, j) = cv::Vec3b(hValueB, hValueG, hValueR);
		}
	}
	return res;
}

cv::Vec3b Brightness2Color(uchar brightness) {
	auto res = cv::Vec3b();
	res[2] = brightness;
	res[1] = (uchar)(brightness * ((uchar)255 - brightness));
	res[0] = (uchar)255 - brightness;
	return res;
}

void Plot(std::vector<int> data,std::string name) {
	int width = 800;  
	int height = 400; 
	cv::Mat graph(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
	int maxValue = *std::max_element(data.begin(), data.end());

	for (int i = 0; i < data.size() - 1; i++) {
		int x1 = i * (width / (data.size() - 1));
		int x2 = (i + 1) * (width / (data.size() - 1));
		int y1 = height - (data[i] * height / maxValue);
		int y2 = height - (data[i + 1] * height / maxValue);
		cv::line(graph, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 0), 2);
	}
	cv::imshow(name, graph);
}

std::vector<int> Array2Vector(int array[],int len) {
	std::vector<int> res;
	for (int i = 0; i < len; i++)
		res.push_back(array[i]);
	return res;
}

void PrintColor(cv::Vec3b pixel) {
	std::cout << "Red = " << (int)pixel[2] << " Green = " << (int)pixel[1] << " Blue = " << (int)pixel[0] << std::endl;
}