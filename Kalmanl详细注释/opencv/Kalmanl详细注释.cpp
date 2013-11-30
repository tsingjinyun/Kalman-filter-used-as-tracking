/*
cvRandInit()
初始化CvRandState数据结构,可以选定随机分布的种类,并给定它种子,有两种情形
cvRandInit(CvRandState数据结构,随机上界,随机下界,均匀分布参数,64bits种子的数字)---- CV_RAND_UNI 指定为均匀分布类型
cvRandInit(CvRandState数据结构,平均数,标准偏差,常态分布参数,64bits种子的数字)---- CV_RAND_NORMAL


cvRandSetRange()
修改CvRandState数据结构的参数内容,均匀分布的话可以每个信道的上下界常态分布可以修改每个通道的平均数,标准偏差.
cvRandSetRange(CvRandState数据结构,均匀分布上界,均匀分布下界,目标信道数据)
cvRandSetRange(CvRandState数据结构,常态分布平均数,常态分布标准偏差,目标信道数据)

cvRand()
将CvMat或IplImage数据结构随机化,用被设定过的CvRandState数据结构来随机.
cvRand(CvRandState数据结构,CvMat或IplImage数据结构)

cvbRand()
将一维数组随机化,可以设定随机的长度
cvbRand(RandState数据结构,Float型别数组名,随机的长度);
*/


//cvKalman结构详细说明
/*
typedef struct CvKalman
{
    int MP;                     /* 测量向量维数 */
    int DP;                     /* 状态向量维数 */
    int CP;                     /* 控制向量维数 */

    /* 向后兼容字段 */
		#if 1
			float* PosterState;         /* =state_pre->data.fl */
			float* PriorState;          /* =state_post->data.fl */
			float* DynamMatr;           /* =transition_matrix->data.fl */
			float* MeasurementMatr;     /* =measurement_matrix->data.fl */
			float* MNCovariance;        /* =measurement_noise_cov->data.fl */
			float* PNCovariance;        /* =process_noise_cov->data.fl */
			float* KalmGainMatr;        /* =gain->data.fl */
			float* PriorErrorCovariance;/* =error_cov_pre->data.fl */
			float* PosterErrorCovariance;/* =error_cov_post->data.fl */
			float* Temp1;               /* temp1->data.fl */
			float* Temp2;               /* temp2->data.fl */
		#endif

    CvMat* state_pre;           /* 预测状态 (x'(k)): 
                                    x(k)=A*x(k-1)+B*u(k) */
    CvMat* state_post;          /* 矫正状态 (x(k)):
                                    x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) */
    CvMat* transition_matrix;   /* 状态传递矩阵 state transition matrix (A) */
    CvMat* control_matrix;      /* 控制矩阵 control matrix (B)
                                   (如果没有控制，则不使用它)*/
    CvMat* measurement_matrix;  /* 测量矩阵 measurement matrix (H) */
    CvMat* process_noise_cov;   /* 过程噪声协方差矩阵
                                        process noise covariance matrix (Q) */
    CvMat* measurement_noise_cov; /* 测量噪声协方差矩阵
                                          measurement noise covariance matrix (R) */
    CvMat* error_cov_pre;       /* 先验误差计协方差矩阵
                                        priori error estimate covariance matrix (P'(k)):
                                     P'(k)=A*P(k-1)*At + Q)*/
    CvMat* gain;                /* Kalman 增益矩阵 gain matrix (K(k)):
                                    K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)*/
    CvMat* error_cov_post;      /* 后验错误估计协方差矩阵
                                        posteriori error estimate covariance matrix (P(k)):
                                     P(k)=(I-K(k)*H)*P'(k) */
    CvMat* temp1;               /* 临时矩阵 temporary matrices */
    CvMat* temp2;
    CvMat* temp3;
    CvMat* temp4;
    CvMat* temp5;
}CvKalman;

*/



#include "StdAfx.h"
#include "cv.h"
#include "highgui.h"
#include <math.h>



int main(int argc, char** argv)
{
 
	cvNamedWindow( "Kalman", 1 );//创建窗口，当为的时候，表示窗口大小自动设定
	CvRandState rng;
	cvRandInit( &rng, 0, 1, -1, CV_RAND_UNI );/* CV_RAND_UNI 指定为均匀分布类型、随机数种子为-1----------后面又改为高斯分布 */

	IplImage* img = cvCreateImage( cvSize(500,500), 8, 3 );
	CvKalman* kalman = cvCreateKalman( 2, 1, 0 );/*状态向量为维，观测向量为维，无激励输入维*/

	// State is phi, delta_phi - angle and angular velocity
	// Initialize with random guess
	CvMat* x_k = cvCreateMat( 2, 1, CV_32FC1 );/*创建行列、元素类型为CV_32FC1，元素为位单通道浮点类型矩阵。*/
	cvRandSetRange( &rng, 0, 0.1, 0 );/*设置随机数范围，随机数服从正态分布，均值为，均方差为.1，通道个数为*/
	rng.disttype = CV_RAND_NORMAL;
	cvRand( &rng, x_k ); /*随机填充数组*/

	// Process noise
	CvMat* w_k = cvCreateMat( 2, 1, CV_32FC1 );

	// Measurements, only one parameter for angle
	CvMat* z_k = cvCreateMat( 1, 1, CV_32FC1 );/*定义观测变量*/
	cvZero( z_k ); /*矩阵置零*/

	// Transition matrix F describes model parameters at and k and k+1
	const float F[] = { 1, 1, 0, 1 }; /*状态转移矩阵*/
	memcpy( kalman->transition_matrix->data.fl, F, sizeof(F));
	/*初始化转移矩阵，行列，具体见CvKalman* kalman = cvCreateKalman( 2, 1, 0 );*/

	// Initialize other Kalman parameters
	cvSetIdentity( kalman->measurement_matrix, cvRealScalar(1) );/*观测矩阵*/
	cvSetIdentity( kalman->process_noise_cov, cvRealScalar(1e-5) );/*过程噪声*/
	cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(1e-1) );/*观测噪声*/
	cvSetIdentity( kalman->error_cov_post, cvRealScalar(1) );/*后验误差协方差*/

	// Choose random initial state
	cvRand( &rng, kalman->state_post );/*初始化状态向量*/

	// Make colors
	CvScalar yellow = CV_RGB(255,255,0);/*依次为红绿蓝三色*/
	CvScalar white = CV_RGB(255,255,255);
	CvScalar red = CV_RGB(255,0,0);

	while( 1 )
	{
		// Predict point position
		const CvMat* y_k = cvKalmanPredict( kalman, 0 );/*激励项输入为*/

		// Generate Measurement (z_k)
		cvRandSetRange( &rng, 0, sqrt( kalman->measurement_noise_cov->data.fl[0] ), 0 );/*设置观测噪声*/	
		cvRand( &rng, z_k );
		cvMatMulAdd( kalman->measurement_matrix, x_k, z_k, z_k );

		// Update Kalman filter state
		cvKalmanCorrect( kalman, z_k );

		// Apply the transition matrix F and apply "process noise" w_k
		cvRandSetRange( &rng, 0, sqrt( kalman->process_noise_cov->data.fl[0] ), 0 );/*设置正态分布过程噪声*/
		cvRand( &rng, w_k );
		cvMatMulAdd( kalman->transition_matrix, x_k, w_k, x_k );

		// Plot Points
		cvZero( img );/*创建图像*/
		// Yellow is observed state 黄色是观测值
		//cvCircle(IntPtr, Point, Int32, MCvScalar, Int32, LINE_TYPE, Int32)
		//对应于下列其中，shift为数据精度
		//cvCircle(img, center, radius, color, thickness, lineType, shift)
		//绘制或填充一个给定圆心和半径的圆
		cvCircle( img, 
			cvPoint( cvRound(img->width/2 + img->width/3*cos(z_k->data.fl[0])),
			cvRound( img->height/2 - img->width/3*sin(z_k->data.fl[0])) ), 
			4, yellow );
		// White is the predicted state via the filter
		cvCircle( img, 
			cvPoint( cvRound(img->width/2 + img->width/3*cos(y_k->data.fl[0])),
			cvRound( img->height/2 - img->width/3*sin(y_k->data.fl[0])) ), 
			4, white, 2 );
		// Red is the real state
		cvCircle( img, 
			cvPoint( cvRound(img->width/2 + img->width/3*cos(x_k->data.fl[0])),
			cvRound( img->height/2 - img->width/3*sin(x_k->data.fl[0])) ),
			4, red );
		CvFont font;
		cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,0.5f,0.5f,0,1,8);
		cvPutText(img,"Yellow:observe",cvPoint(0,20),&font,cvScalar(0,0,255));
		cvPutText(img,"While:predict",cvPoint(0,40),&font,cvScalar(0,0,255));
		cvPutText(img,"Red:real",cvPoint(0,60),&font,cvScalar(0,0,255));
		cvPutText(img,"Press Esc to Exit...",cvPoint(0,80),&font,cvScalar(255,255,255));
		cvShowImage( "Kalman", img );		

		// Exit on esc key
		if(cvWaitKey(100) == 27) 
			break;
	}
	cvReleaseImage(&img);/*释放图像*/
	cvReleaseKalman(&kalman);/*释放kalman滤波对象*/
	cvDestroyAllWindows();/*释放所有窗口*/
}
