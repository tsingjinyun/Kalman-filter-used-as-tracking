/* 
   Tracking of rotating point.
   Rotation speed is constant.
   Both state and measurements vectors are 1D (a point angle),
   Measurement is the real point angle + gaussian noise.
   The real and the estimated points are connected with yellow line segment,
   the real and the measured points are connected with red line segment.
   (if Kalman filter works correctly,
    the yellow segment should be shorter than the red one).
   Pressing any key (except ESC) will reset the tracking with a different speed.
   Pressing ESC will stop the program.
*/

#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <math.h>
#endif

int main(int argc, char** argv)
{
    const float A[] = { 1, 1, 0, 1 };//状态转移矩阵
    
    IplImage* img = cvCreateImage( cvSize(500,500), 8, 3 );//创建显示所用的图像
    CvKalman* kalman = cvCreateKalman( 2, 1, 0 );//创建cvKalman数据结构，状态向量为2维，观测向量为1维，无激励输入维
    CvMat* state = cvCreateMat( 2, 1, CV_32FC1 ); //(phi, delta_phi) 定义了状态变量
    CvMat* process_noise = cvCreateMat( 2, 1, CV_32FC1 );// 创建两行一列CV_32FC1的单通道浮点型矩阵
    CvMat* measurement = cvCreateMat( 1, 1, CV_32FC1 ); //定义观测变量
    CvRNG rng = cvRNG(-1);//初始化一个随机序列函数
    char code = -1;

    cvZero( measurement );//观测变量矩阵置零
    cvNamedWindow( "Kalman", 1 );

    for(;;)
    {		//用均匀分布或者正态分布的随机数填充输出数组state
        cvRandArr( &rng, state, CV_RAND_NORMAL, cvRealScalar(0), cvRealScalar(0.1) );//状态state
        memcpy( kalman->transition_matrix->data.fl, A, sizeof(A));//初始化状态转移F矩阵
        
        //cvSetIdentity()用法：把数组中除了行数与列数相等以外的所有元素的值都设置为0；行数与列数相等的元素的值都设置为1
        //我们将(第一个前假象阶段的)后验状态初始化为一个随机值
        cvSetIdentity( kalman->measurement_matrix, cvRealScalar(1) );//观测矩阵H
        cvSetIdentity( kalman->process_noise_cov, cvRealScalar(1e-5) );//过程噪声Q
        cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(1e-1) );//观测噪声R 
        cvSetIdentity( kalman->error_cov_post, cvRealScalar(1));//后验误差协方差
        cvRandArr( &rng, kalman->state_post, CV_RAND_NORMAL, cvRealScalar(0), cvRealScalar(0.1) );//校正状态
        
        //在时机动态系统上开始预测
        
        for(;;)
        {
            #define calc_point(angle)                                      \
                cvPoint( cvRound(img->width/2 + img->width/3*cos(angle)),  \
                         cvRound(img->height/2 - img->width/3*sin(angle))) 

            float state_angle = state->data.fl[0];
            CvPoint state_pt = calc_point(state_angle);
            
            const CvMat* prediction = cvKalmanPredict( kalman, 0 );//计算下一个时间点的预期值，激励项输入为0
            float predict_angle = prediction->data.fl[0];
            CvPoint predict_pt = calc_point(predict_angle);
            
            float measurement_angle;
            CvPoint measurement_pt;

            cvRandArr( &rng, measurement, CV_RAND_NORMAL, cvRealScalar(0),
                       cvRealScalar(sqrt(kalman->measurement_noise_cov->data.fl[0])) );

            /* generate measurement */
            cvMatMulAdd( kalman->measurement_matrix, state, measurement, measurement );
            //cvMatMulAdd(src1,src2,src3,dst)就是实现dist=src1*src2+src3; 

            measurement_angle = measurement->data.fl[0];
            measurement_pt = calc_point(measurement_angle);
            
            //调用Kalman滤波器并赋予其最新的测量值，接下来就是产生过程噪声，然后对状态乘以传递矩阵F完成一次迭代并加上我们产生的过程噪声
            /* plot points */
            #define draw_cross( center, color, d )                                 \
                cvLine( img, cvPoint( center.x - d, center.y - d ),                \
                             cvPoint( center.x + d, center.y + d ), color, 1, CV_AA, 0); \
                cvLine( img, cvPoint( center.x + d, center.y - d ),                \
                             cvPoint( center.x - d, center.y + d ), color, 1, CV_AA, 0 )

            cvZero( img );
            //使用上面宏定义的函数
            draw_cross( state_pt, CV_RGB(255,255,255), 3 );//白色，状态点
            draw_cross( measurement_pt, CV_RGB(255,0,0), 3 );//红色，测量点
            draw_cross( predict_pt, CV_RGB(0,255,0), 3 );//绿色，估计点
            cvLine( img, state_pt, measurement_pt, CV_RGB(255,0,0), 3, CV_AA, 0 );
            cvLine( img, state_pt, predict_pt, CV_RGB(255,255,0), 3, CV_AA, 0 );
            
            cvKalmanCorrect( kalman, measurement );//校正新的测量值

            cvRandArr( &rng, process_noise, CV_RAND_NORMAL, cvRealScalar(0),
                       cvRealScalar(sqrt(kalman->process_noise_cov->data.fl[0])));//设置正态分布过程噪声
            cvMatMulAdd( kalman->transition_matrix, state, process_noise, state );

            cvShowImage( "Kalman", img );
			//当按键按下时，开始新的循环，初始矩阵可能会改变，所以移动速率会改变
            code = (char) cvWaitKey( 100 );
            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }
    
    cvDestroyWindow("Kalman");

    return 0;
}

#ifdef _EiC
main(1, "kalman.c");
#endif
