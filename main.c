#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>

#define MAX_TRAIN_SIZE 50
#define IP_SIZE 3
#define OP_SIZE 1
#define LEARNING_RATE 0.2
#define EPOCHS 500
#define THRESHHOLD 0.6

double traindata[MAX_TRAIN_SIZE][MAX_TRAIN_SIZE];
double testdata[MAX_TRAIN_SIZE][MAX_TRAIN_SIZE];

struct node {
    double fin;
    double fnet;
    double delta;
};

double Activate(double x) {
    return (double)1/(1+exp(-1*x));
}

double derivative(double x) {
    return (double)exp(-1*x)/((1+exp(-1*x))*(1+exp(-1*x)));
} 

int main(void) {

    FILE *trainfp = fopen("train.csv", "r");
    char buff[1024];

    if(!trainfp) {
        printf("Can't open file\n");
        return 1;
    }

    int row_count = 0, col_count = 0;
    while(fgets(buff, 1024, trainfp)) {

        col_count = 0;

        char* fieldData = strtok(buff, ",");
        while(fieldData) {
            double cellData = atoi(fieldData);
            traindata[row_count][col_count++] = cellData;
            fieldData = strtok(NULL, ",");
        }

        row_count++;
    }

    fclose(trainfp);

    double ip[row_count][IP_SIZE];
    double op[row_count][OP_SIZE];

    for(int i=0;i<row_count;i++) {
        int k = 0;
        for(int j=0;j<col_count;j++) {
            if(j==col_count-1 && k<OP_SIZE) {
                op[i][k++] = traindata[i][j];
            } else {
                ip[i][j] = traindata[i][j];
            }
            printf("%lf ", traindata[i][j]);
        }
        printf("\n");
    }

    // Inititalise Weights
    double wij[3][3], wjk[1][3];
    for(int i=0;i<3;i++) {
        for(int j=0;j<3;j++) {
            wij[i][j] = (double)rand() / (double)RAND_MAX;
        }
        wjk[0][i] = (double)rand() / (double)RAND_MAX;
    }

    // Initialise Bias
    // between Input and Hidden Layer
    double bias[IP_SIZE][2];
    for(int i=0;i<IP_SIZE;i++) {
        bias[i][0] = 0;
        bias[i][1] = (double)rand() / (double)RAND_MAX;
    }
    
    // Between Hidden Layer and Output Layer
    double outputBias[2];
    outputBias[0] = 0;
    outputBias[1] = (double)rand() / (double)RAND_MAX;

    // Initialise
    struct node Nodes[2][3];
    struct node output;

    int iterations = EPOCHS;
    while(iterations>0) {
    
        for(int t=0;t<row_count;t++) {
            // Feed Forward
            // Initialise Input Layer
            for(int i=0;i<3;i++) {
                Nodes[0][i].fin = ip[t][i];
                Nodes[0][i].fnet = ip[t][i];
            }

            // Summation Part of Neuron between Input and Hidden Layer
            for(int j=0;j<3;j++) {
                double sum = bias[j][0]*bias[j][1];
                for(int i=0;i<3;i++) {
                    sum += wij[i][j]*Nodes[0][i].fnet;
                }
                Nodes[1][j].fin = sum;
                Nodes[1][j].fnet = Activate(Nodes[1][j].fin);
            }

            // Summation Part of Neuron between Hidden Layer and Output Layer
            double sum = outputBias[0]*outputBias[1];
            for(int j=0;j<3;j++) {
                sum += Nodes[1][j].fnet * wjk[0][j];
            }
            output.fin = sum;
            output.fnet = Activate(output.fin);
            
            double result = (output.fnet>=THRESHHOLD) ? 1 : 0;
            printf("%0.1lf ", result);

            // Back Propagation
            output.delta = (op[t][0] - output.fnet) * derivative(output.fin);

            // Adjust weights 

            // between op layer and hidden layer
            for(int i=0;i<3;i++) {
                wjk[0][i] += LEARNING_RATE*output.delta*Nodes[0][i].fnet;
            }
            outputBias[1] += LEARNING_RATE*output.delta;

            // Between Hidden Layer and input layer
            
            // Calculate Delta for each node in hidden layer
            for(int i=0;i<IP_SIZE;i++) {
                double sum = 0;
                for(int j=0;j<IP_SIZE;j++) {
                    sum += output.delta*wjk[0][j];
                }
                Nodes[0][i].delta = sum*derivative(Nodes[0][i].fin);
                for(int j=0;j<IP_SIZE;j++) {
                    wij[i][j] += LEARNING_RATE*Nodes[0][j].delta*ip[t][i];
                }
                bias[i][0] += LEARNING_RATE*Nodes[0][i].delta;
            }
        }
        printf("\n");

        iterations--;
    }
    printf("\n");


    /*====================================Model Trained======================================*/


    // Predict Function
    FILE *testfp = fopen("test.csv", "r");

    if(!testfp) {
        printf("Can't open file\n");
        return 1;
    }

    row_count = 0;
    col_count = 0;
    while(fgets(buff, 1024, testfp)) {

        col_count = 0;

        char* fieldData = strtok(buff, ",");
        while(fieldData) {
            double cellData = atoi(fieldData);
            testdata[row_count][col_count++] = cellData;
            fieldData = strtok(NULL, ",");
        }

        row_count++;
    }

    fclose(testfp);

    double testip[row_count][IP_SIZE];
    double testop[row_count][OP_SIZE];

    for(int i=0;i<row_count;i++) {
        int k = 0;
        for(int j=0;j<col_count;j++) {
            if(j==col_count-1 && k<OP_SIZE) {
                testop[i][k++] = testdata[i][j];
            } else {
                testip[i][j] = testdata[i][j];
            }
        }
    }

    // Initialise Input Neurons
    struct node ipNodes[2][3];
    struct node ans;

    int correct = 0;

    for(int t=0;t<row_count;t++) {
        
        // Initialise Input Layer
        for(int i=0;i<3;i++) {
            ipNodes[0][i].fin = testip[t][i];
            ipNodes[0][i].fnet = testip[t][i];
        }

        // Summation Part of Neuron between Input and Hidden Layer
        for(int j=0;j<3;j++) {
            double sum = bias[j][0]*bias[j][1];
            for(int i=0;i<3;i++) {
                sum += wij[i][j]*ipNodes[0][i].fnet;
            }
            ipNodes[1][j].fin = sum;
            ipNodes[1][j].fnet = Activate(ipNodes[1][j].fin);
        }

        // Summation Part of Neuron between Hidden Layer and Output Layer
        double sum = outputBias[0]*outputBias[1];
        for(int j=0;j<3;j++) {
            sum += ipNodes[1][j].fnet * wjk[0][j];
        }
        ans.fin = sum;
        ans.fnet = Activate(ans.fin);
        
        double result = (ans.fnet>=THRESHHOLD) ? 1 : 0;
        printf("%0.1lf\n", result);

        // For calculating the accuracy
        if(result==testop[t][0]) correct++;
    }

    printf("Accuracy: %.2lf\n", (double)correct*100/row_count);
    return 0;
}
