//Huseyin Uzun - Perceptron with OpenCV Project.
//This perceptron project uses OpenCV - version 4.7.0.
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int const MAX_ITER = 10;
float const LEARNING_RATE = 0.1;
int const NUM_INST = 700;
int const theta = 0;

int calculateOutput(int theta, float* weights, float* attributes, float bias) {
    float sum = 0;
    for (int i = 0; i < NUM_INST / 2; i++) { 
        sum += attributes[i] * weights[i];
    }
    sum += bias;
    return (sum >= theta) ? 1 : 0;
}

int main()
{
    Mat image1 = imread("stone.bmp");
    Mat image2 = imread("wood.bmp");
    Mat image3 = imread("metal.bmp");
    Mat image4 = imread("water.bmp");
    Mat image5 = imread("granit.bmp");
    Mat image6 = imread("diamond.bmp");
    Mat image7 = imread("gold.bmp");


    if (image1.empty() || image2.empty() || image3.empty() || image4.empty() || image5.empty() || image6.empty() || image7.empty())
    {
        cout << "Error - Failed to open: .bmp\n";
        return 1;
    }

    int outputs[NUM_INST];
    float rclass[NUM_INST][NUM_INST / 2];

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec3b colour = image1.at<Vec3b>(j, i);
            rclass[i][j] = 0.3 * colour[2] + 0.11 * colour[0] + 0.59 * colour[1];
        }
        outputs[i] = 0;
    }

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec3b colour = image2.at<Vec3b>(j, i);
            rclass[100 + i][j] = 0.3 * colour[2] + 0.11 * colour[0] + 0.59 * colour[1];
        }
        outputs[100 + i] = 1;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec3b colour = image3.at<Vec3b>(j, i);
            rclass[200 + i][j] = 0.3 * colour[2] + 0.11 * colour[0] + 0.59 * colour[1];
        }
        outputs[200 + i] = 1;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec3b colour = image4.at<Vec3b>(j, i);
            rclass[300 + i][j] = 0.3 * colour[2] + 0.11 * colour[0] + 0.59 * colour[1];
        }
        outputs[300 + i] = 1;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec3b colour = image5.at<Vec3b>(j, i);
            rclass[400 + i][j] = 0.3 * colour[2] + 0.11 * colour[0] + 0.59 * colour[1];
        }
        outputs[400 + i] = 1;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec3b colour = image6.at<Vec3b>(j, i);
            rclass[500 + i][j] = 0.3 * colour[2] + 0.11 * colour[0] + 0.59 * colour[1];
        }
        outputs[500 + i] = 1;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            Vec3b colour = image7.at<Vec3b>(j, i);
            rclass[600 + i][j] = 0.3 * colour[2] + 0.11 * colour[0] + 0.59 * colour[1];
        }
        outputs[600 + i] = 1;
    }




    float weights[NUM_INST / 2];
    float localError, globalError;
    int p, iteration, output;
    for (int i = 0; i < NUM_INST / 2; i++) {
        weights[i] = ((float)rand() / (float)RAND_MAX);
    }

    float bias = ((float)rand() / (float)RAND_MAX);
    iteration = 0;
    do {
        iteration++;
        globalError = 0;
      
        for (p = 0; p < NUM_INST; p++) {
            float row[NUM_INST / 2];
            for (int i = 0; i < NUM_INST / 2; i++) {
                row[i] = rclass[p][i];
            }
            // calculate predicted class
            output = calculateOutput(theta, weights, row, bias);
       
            localError = outputs[p] - output;
    
            for (int w = 0; w < NUM_INST / 2; w++) {
                weights[w] += LEARNING_RATE * localError * rclass[p][w];
            }
            bias += LEARNING_RATE * localError;
          
            globalError += (localError * localError);
        }
        // Root Mean Squared Error 
        cout << "Iteration " << iteration << " : RMSE = " << sqrt(globalError / double(NUM_INST)) << "\n";
    } while (globalError != 0 && iteration <= MAX_ITER);

    cout << "\n =======\nDecision boundary equation:\n";
    for (int i = 0; i < NUM_INST / 2; i++) {
        cout << weights[i] << " * p" << i << " + ";
    }

    float unknown[NUM_INST / 2];
    int r = rand() % NUM_INST + 1; // test on training data - bad practice
    for (int t = 0; t < NUM_INST / 2; t++) {
        unknown[t] = rclass[r][t];
    }

    //random pixel for test 
    output = calculateOutput(theta, weights, unknown, bias);
    cout << "\n\n =======\nRandom pixel row:";
    cout << "class = " << output;
    if (r < NUM_INST / 2) {
        cout << " when must be 0" << endl;
    }
    else {
        cout << " when must be 1" << endl;
    }

    return 0;
}
