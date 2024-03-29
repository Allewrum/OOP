#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

class ImageProcessor {
private:
    cv::Mat originalImage;  // Исходное изображение
    cv::Mat processedImage; // Обработанное изображение
    cv::Mat gradientX;       // Горизонтальный градиент
    cv::Mat gradientY;       // Вертикальный градиент
public:

    ImageProcessor(const string& filePath);
    ~ImageProcessor();

    void SobelFilter();
    void LaplacianFilter();
    void PrewittFilter();
    void RobertsFilter();
    void WallisFilter();
    void CannyFilter();


    void showImage(const string& windowName, const cv::Mat& image);
};

void ImageProcessor::showImage(const string& windowName, const cv::Mat& image) {
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);
    cv::waitKey(0);
}


ImageProcessor::ImageProcessor(const string& filePath) {
    originalImage = imread(filePath);
    if (originalImage.empty()) {
        cerr << "Error: Unable to load the image!" << endl;
        exit(EXIT_FAILURE);
        // обработка ошибки загрузки изображения
    }
}

ImageProcessor::~ImageProcessor() {
    // Release any resources if needed
}

// Применение фильтра Собеля к изображению с использованием заданного ядра
void applySobelFilter(const int kernel[3][3], const cv::Mat& inputImage, cv::Mat& outputImage) {
    outputImage = cv::Mat::zeros(inputImage.size(), CV_16S);

    for (int y = 1; y < inputImage.rows - 1; ++y) {
        for (int x = 1; x < inputImage.cols - 1; ++x) {
            int sum = 0;

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    sum += kernel[i + 1][j + 1] * inputImage.at<uchar>(y + i, x + j);
                }
            }

            outputImage.at<short>(y, x) = sum;
        }
    }
}

// Преобразование изображения с градиентами в изображение 8-бит
cv::Mat ConvertTo8bitImage(const cv::Mat& inputImage) {
    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), CV_8U);

    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            int value = inputImage.at<int>(y, x);
            value = min(255, max(0, value)); // Ограничиваем значение в диапазоне 0-255
            outputImage.at<uchar>(y, x) = static_cast<uchar>(value);
        }
    }

    return outputImage;
}

// Сложение взвешенных изображений
void addWeightedImages(const cv::Mat& image1, const cv::Mat& image2, cv::Mat& resultImage) {
    resultImage = cv::Mat::zeros(image1.size(), CV_8U);

    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            int sum = image1.at<uchar>(y, x) / 2 + image2.at<uchar>(y, x) / 2;
            resultImage.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }
}


 
void ImageProcessor::SobelFilter() {
    // Клонируем оригинальное изображение
    processedImage = originalImage.clone();

    // Преобразуем изображение в оттенки серого
    cv::cvtColor(originalImage, processedImage, COLOR_BGR2GRAY);

    // Ядро оператора Собеля для вычисления градиентов по x и y
    int sobelKernelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobelKernelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // Применяем фильтр Собеля
    applySobelFilter(sobelKernelX, processedImage, gradientX);
    applySobelFilter(sobelKernelY, processedImage, gradientY);

    // Преобразование обратно в изображение 8-бит
    ConvertTo8bitImage(gradientX);
    ConvertTo8bitImage(gradientY);

    // Сложение горизонтальных и вертикальных градиентов
    addWeightedImages(gradientX, gradientY, processedImage);

    // Отображаем обработанное изображение
    cv::imshow("SobelFilter", processedImage);
    cv::waitKey(0);

    // Отображаем оригинальное изображение
    cv::imshow("Original", originalImage);
    cv::waitKey(0);
}



// Применение фильтра Лапласа к изображению
void applyLaplacianFilter(const cv::Mat& inputImage, cv::Mat& outputImage) {
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int sum = -4 * inputImage.at<uchar>(y, x);
            sum += inputImage.at<uchar>(y - 1, x) + inputImage.at<uchar>(y + 1, x)
                + inputImage.at<uchar>(y, x - 1) + inputImage.at<uchar>(y, x + 1);

            outputImage.at<short>(y, x) = sum;
        }
    }
}

// Преобразование изображения с градиентами в изображение 8-бит
void convertTo8bitImage(const cv::Mat& inputImage, cv::Mat& outputImage) {
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int value = inputImage.at<short>(y, x);
            value = std::min(255, std::max(0, value)); // Ограничиваем значение в диапазоне 0-255
            outputImage.at<uchar>(y, x) = static_cast<uchar>(value);
        }
    }
}
void ImageProcessor::LaplacianFilter() {
    // 1. Клонирование оригинального изображения
    processedImage = originalImage.clone();

    // 2. Преобразование изображения в оттенки серого, если оно не одноканальное
    if (processedImage.channels() > 1) {
        cvtColor(originalImage, processedImage, COLOR_BGR2GRAY);
    }

    // 3. Применение фильтра Лапласа
    cv::Mat laplacianImage(processedImage.size(), CV_16S);

    applyLaplacianFilter(processedImage, laplacianImage);

    // 4. Преобразование обратно в изображение 8-бит
    convertTo8bitImage(laplacianImage, processedImage);

    // 5. Отображение изображения
    cv::imshow("Laplacian Filter", processedImage);
    cv::waitKey(0);
}




void ImageProcessor::PrewittFilter() {
    // Преобразование изображения в оттенки серого
    cv::cvtColor(originalImage, processedImage, cv::COLOR_BGR2GRAY);

    // Получаем размеры изображения
    int rows = processedImage.rows;
    int cols = processedImage.cols;

    // Создаем матрицу для результата
    cv::Mat resultImage(rows, cols, CV_8UC1, cv::Scalar(0));

    // Обрабатываем каждый пиксель входного изображения
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            uchar val = processedImage.at<uchar>(i, j);

            // Применяем операцию Прюитта к текущему пикселю
            uchar left = processedImage.at<uchar>(i - 1, j - 1);
            uchar right = processedImage.at<uchar>(i + 1, j + 1);

            if ((left != 0 && right != 0) || (left == 0 && right == 0)) {
                val = left ^ right;
            }

            // Записываем результат обработки в выходную матрицу
            resultImage.at<uchar>(i, j) = val;
        }
    }

    // Отображаем или сохраняем результат
    cv::imshow("Prewitt Filter", resultImage);
    cv::waitKey(0);
}


void ImageProcessor::RobertsFilter() {

    processedImage = originalImage.clone();

    // преобразование изображения в оттенки серого
    cv::cvtColor(originalImage, processedImage, cv::COLOR_BGR2GRAY);

    // применение фильтров Робертса
    cv::Mat kernelX = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
    cv::Mat kernelY = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);

    cv::Mat gradientX, gradientY;
    cv::filter2D(processedImage, gradientX, CV_16S, kernelX);
    cv::filter2D(processedImage, gradientY, CV_16S, kernelY);

    // находим абсолютные значения градиентов
    gradientX = cv::abs(gradientX);
    gradientY = cv::abs(gradientY);

    // сложение градиентов
    addWeighted(gradientX, 0.5, gradientY, 0.5, 0, processedImage);

    // преобразование обратно в изображение 8-бит
    convertScaleAbs(processedImage, processedImage);

    // Отображаем или сохраняем изображение
    cv::imshow("Roberts Filter", processedImage);
    cv::waitKey(0);
}


void ImageProcessor::WallisFilter() {
    // Преобразование изображения в оттенки серого
    cv::cvtColor(originalImage, processedImage, cv::COLOR_BGR2GRAY);
    //Преобразование изображения в оттенки серого необходимо для того, чтобы применить оператор Уоллеса к каждому пикселю в изображении независимо от его цвета.

    // Получаем размеры изображения
    int rows = processedImage.rows;
    int cols = processedImage.cols;

    // Создаем матрицу для результата
    cv::Mat resultImage(rows, cols, CV_8UC1, cv::Scalar(0));

    // Обрабатываем каждый пиксель входного изображения
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar val = processedImage.at<uchar>(i, j);

            // Применяем операцию Уоллеса к текущему пикселю
            if (val > 0 && val <= 64) {
                val += 32;
            }
            else if (val >= 96 && val <= 127) {
                val -= 32;
            }

            // Записываем результат обработки в выходную матрицу
            resultImage.at<uchar>(i, j) = val;
        }
    }

    // Отображаем или сохраняем результат
    cv::imshow("Wallis Filter", resultImage);
    cv::waitKey(0);
}


void ImageProcessor::CannyFilter() {
    int kernelSize = 3;
    double lowThreshold = 10;
    double highThreshold = 100;
    //Mat& edges = ; 
    // Apply Gaussian blur to reduce noise
    cv::Mat gaussianBlurred;
    cv::GaussianBlur(originalImage, gaussianBlurred, Size(kernelSize, kernelSize), 0, 0);

    // Calculate gradient magnitude and direction
    Mat gradMag, gradDir;
    Sobel(gaussianBlurred, gradMag, CV_64FC1, 1, 0, kernelSize);
    Sobel(gaussianBlurred, gradDir, CV_64FC1, 0, 1, kernelSize);

    // Find non-zero pixels in the gradient magnitude matrix
    vector<Vec2d> nonZeroPixels;
    findNonZero(gradMag.reshape(1), nonZeroPixels);

    // Create output matrix
    cv::Mat edges = Mat::zeros(gradMag.size(), CV_8UC1);

    // Loop through each pixel and check if it is above the low threshold
    for (int r = 0; r < gradMag.rows; r++)
    {
        for (int c = 0; c < gradMag.cols; c++)
        {
            Vec2d grad = gradMag.at<Vec2d>(r, c);
            if (grad.val[0] > lowThreshold)
            {
                // Check if the pixel is also above the high threshold
                bool isAboveHighThreshold = false;
                for (int i = 0; i < nonZeroPixels.size(); i++)
                {
                    if (grad.val[0] > nonZeroPixels[i].val[0])
                    {
                        isAboveHighThreshold = true;
                        break;
                    }
                }

                if (isAboveHighThreshold)
                {
                    edges.at<uint8_t>(r, c) = 255;
                }
            }
        }
    }

    cv::namedWindow("Canny Edges", cv::WINDOW_AUTOSIZE);
    cv::imshow("Canny Edges", edges);
    cv::waitKey(0);
}



class ContourDetector {
private:
    cv::Mat inputImage_;

    void preprocessImage(const cv::Mat& inputImage, cv::Mat& outputImage) {
        cv::Mat grayImage;
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

        cv::Mat blurredImage;
        cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);

        cv::Mat thresholdedImage;
        cv::threshold(blurredImage, thresholdedImage, 100, 255, cv::THRESH_BINARY);

        outputImage = thresholdedImage;
    }
public:
    ContourDetector(const std::string& imagePath) {
        inputImage_ = cv::imread(imagePath);
        if (inputImage_.empty()) {
            cerr << "Error loading image from path: " << imagePath << endl;
        }
    }

    void detectAndDrawRectangles() {
        cv::Mat filteredImage;
        preprocessImage(inputImage_, filteredImage);

        std::vector<std::vector<cv::Point>> contours;
        findContours(filteredImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Создаем копию изображения, чтобы не изменять оригинал
        cv::Mat resultImage = inputImage_.clone();

        for (const auto& contour : contours) {
            cv::Rect boundingRect = cv::boundingRect(contour);
            cv::rectangle(resultImage, boundingRect, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Contours as Rectangles", resultImage);
        cv::waitKey(0);
    }

    void detectAndDrawSegments() {
        cv::Mat filteredImage;
        preprocessImage(inputImage_, filteredImage);

        std::vector<std::vector<cv::Point>> contours;
        findContours(filteredImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Создаем копию изображения, чтобы не изменять оригинал
        cv::Mat resultImage = inputImage_.clone();

        for (const auto& contour : contours) {
            std::vector<cv::Vec4i> hierarchy;
            std::vector<std::vector<cv::Point>> contoursPoly(contours.size());
            cv::approxPolyDP(contour, contoursPoly[0], 3, true);

            cv::polylines(resultImage, contoursPoly, true, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Contours as Segments", resultImage);
        cv::waitKey(0);
    }
};



int main() {
    // Создаем объект класса ImageProcessor, передавая ему путь к изображению
    ImageProcessor imageProcessor("C:/Users/nikit/Desktop/1203_Rome.jpg");

    imageProcessor.LaplacianFilter();
    imageProcessor.CannyFilter();
    imageProcessor.PrewittFilter();
    imageProcessor.SobelFilter();
    imageProcessor.RobertsFilter();
    imageProcessor.WallisFilter();

    ContourDetector contourDetector("C:/Users/nikit/Desktop/cat.jpg");

    // Detect and draw contours as rectangles
    contourDetector.detectAndDrawRectangles();

    // Detect and draw contours as segments
    contourDetector.detectAndDrawSegments();

    char c; cin >> c;
    return 0;
}