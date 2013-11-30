int main (int argc,int ** argv)
　　{
　　	// Initialize Kalman filter object, window, number generator, etc
　　	cvNamedWindow( "Kalman", 1 );//创建窗口，当为的时候，表示窗口大小自动设定
　　	CvRandState rng;
　　	//cvRandInit( &rng, 0, 1, -1, CV_RAND_UNI );/* CV_RAND_UNI 指定为均匀分布类型、随机数种子为-1 *///怎么接下来又改为高斯分布
　　
　　	cvRandInit(&rng,0,0.1,-1,CV_RAND_NORMAL);//自己添加，原函数中先设为均匀分布再改回，觉得没有必要
　　	IplImage* img = cvCreateImage( cvSize(500,500), 8, 3 );
　　	CvKalman* kalman = cvCreateKalman( 2, 1, 0 );/*状态向量为2维，观测向量为1维，无激励输入维*/
　　
　　	// State is phi, delta_phi - angle and angular velocity
　　	// Initialize with random guess
　　	CvMat* x_k = cvCreateMat( 2, 1, CV_32FC1 );/*创建行列、元素类型为CV_32FC1，元素为位单通道浮点类型矩阵。*/
　　	//cvRandSetRange( &rng, 0, 0.1, 0 );/*设置随机数范围，随机数服从正态分布，均值为，均方差为.1，通道个数为*/
　　	//rng.disttype = CV_RAND_NORMAL;
　　	//cvRand( &rng, x_k ); /*随机填充数组*/
　　	x_k->data.fl[0]=0.;//设起始角度为0；//自己修改过的例子中x_k为汽车沿圆周运动的参数，去掉了噪声，在后面与kalman
　　
　　	//滤波器校正后的运动参数对比
　　	x_k->data.fl[1]=0.05f;//设角速度为1,弧度值
　　
　　	// Process noise
　　	CvMat* w_k = cvCreateMat( 2, 1, CV_32FC1 );
　　
　　	// Measurements, only one parameter for angle
　　	CvMat* z_k = cvCreateMat( 1, 1, CV_32FC1 );/*定义观测变量*/
　　	cvZero( z_k ); /*矩阵置零*/
　　
　　	// Transition matrix F describes model parameters at and k and k+1
　　	const float F[] = { 1, 1, 0, 1 }; /*状态转移矩阵*///F 矩阵实际上应该是2*2的矩阵[1,1;0,1]
　　	memcpy( kalman->transition_matrix->data.fl, F, sizeof(F));
　　	/*初始化转移矩阵，行列，具体见CvKalman* kalman = cvCreateKalman( 2, 1, 0 );*/
　　
　　	// Initialize other Kalman parameters
　　	cvSetIdentity( kalman->measurement_matrix, cvRealScalar(1) );/*观测矩阵*///观测矩阵也是1*2 [1,0],因为只能测得汽车的位置
　　	cvSetIdentity( kalman->process_noise_cov, cvRealScalar(1e-5) );/*过程噪声*/
　　	cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(1e-1) );/*观测噪声*/
　　	cvSetIdentity( kalman->error_cov_post, cvRealScalar(1) );/*后验误差协方差*/
　　
　　	// Choose random initial state
　　	cvRand( &rng, kalman->state_post );/*初始化状态向量*/
　　
　　	// Make colors
　　	CvScalar yellow = CV_RGB(255,255,0);/*依次为红绿蓝三色*/
　　	CvScalar white = CV_RGB(255,255,255);
　　	CvScalar red = CV_RGB(255,0,0);
　　
　　	while( 1 ){
　　	 // Predict point position
　　	 const CvMat* y_k = cvKalmanPredict( kalman, 0 );/*激励项输入为*/
　　
　　	 // Generate Measurement (z_k)
　　	 cvRandSetRange( &rng, 0, sqrt( kalman->measurement_noise_cov->data.fl[0] ), 0 );/*设置观测噪声*/ 
　　	 cvRand( &rng, z_k );//此时的z_k并非测量值，只是为观测噪声，其实为理解方便应该设一的独立的变量
　　	 cvMatMulAdd( kalman->measurement_matrix, x_k, z_k, z_k );//z_k为测量值
　　
　　	 // Update Kalman filter state
　　	 cvKalmanCorrect( kalman, z_k );
　　
　　	 //// Apply the transition matrix F and apply "process noise" w_k
　　	 //cvRandSetRange( &rng, 0, sqrt( kalman->process_noise_cov->data.fl[0] ), 0 );/*设置正态分布过程噪声*/
　　	 //cvRand( &rng, w_k );
　　	 //cvMatMulAdd( kalman->transition_matrix, x_k, w_k, x_k );//
　　
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
　　	 CvScalar bule=cvScalar(255,0,0,0);
　　	 cvCircle( img, 
　　	 cvPoint( cvRound(img->width/2 + img->width/3*cos(kalman->state_post->data.fl[0])),
　　	 cvRound( img->height/2 - img->width/3*sin(kalman->state_post->data.fl[0])) ),
　　	 4, bule ,2);
　　
　　	 // Apply the transition matrix F and apply "process noise" w_k
　　	 //cvRandSetRange( &rng, 0, sqrt( kalman->process_noise_cov->data.fl[0] ), 0 );/*设置正态分布过程噪声*/
　　	 //cvRand( &rng, w_k );
　　	 cvMatMulAdd( kalman->transition_matrix, x_k, NULL, x_k );//
　　
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