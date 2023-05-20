//
// Created by caston on 2023/5/20.
// 在这个程序中，我们读取两张图像，进行特征匹配。然后根据匹配得到的特征，计算相机运动以及特征点的位置。这是一个典型的Bundle Adjustment，我们用g2o进行优化。
//
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/concept_check.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
// 理解下鲁棒核的含义
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/se3quat.h>
// 这个文件包含了重投影误差边g2o::EdgeProjectXYZ2UV和位姿顶点g2o::VertexSE3Expmap .
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

int findCorrespondingPoints(const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2);

double cx = 325.5, cy = 253.5, fx = 518.0, fy = 519.0;

int main(int argc, char** argv){
    if(argc != 3){
        cout << "Usage: ba_example img1, img2" << endl;
        exit(1);
    }

    cv::Mat img1 = cv::imread( argv[1] );
    cv::Mat img2 = cv::imread( argv[2] );

    vector<cv::Point2f> pts1, pts2;
    if(findCorrespondingPoints( img1, img2, pts1, pts2 ) == false){
        cout << "匹配点不够" << endl;
        return 0;
    }

    cout << "找到了" << pts1.size() << "组对应特征点。" << endl;

    // 开始配置g2o优化
    // 1-构造g2o中的图
    g2o::SparseOptimizer optimizer;
    // 2-使用linear_solver_eigen线性方程求解器, 6*3的参数
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block;
    unique_ptr<Block::LinearSolverType> linearSolver = make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>> ();
    // 3-定义block_solver
    unique_ptr<Block> solver_ptr(new Block( move(linearSolver) ));
    // 4-定义优化方法，这里选择了Levenberg
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg( move(solver_ptr) );

    optimizer.setAlgorithm( algorithm );
    optimizer.setVerbose( false );

    // 添加两个相机位姿节点
    for ( int i = 0; i < 2; i++ ){
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if(i == 0){
            v->setFixed(true); // 第一个点是固定的，不优化.
        }
        // 预设值为单位Pose， 因为无任何先验知识
        v->setEstimate( g2o::SE3Quat() );
        optimizer.addVertex( v );
    }

    // 很多个特征点的节点
    // 以第一帧为准
    for( size_t i = 0; i < pts1.size(); i++ ){
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        // 不清楚深度，所以设置为1
        double z = 1;
        double x = (pts1[i].x - cx) * z / fx;
        double y = (pts1[i].y - cx) * z / fy;
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x, y, z) );
        optimizer.addVertex(v);
    }

    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0);

    camera->setId(0);
    optimizer.addParameter( camera );

    // 添加边
    // 第一帧
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for(size_t i=0; i < pts1.size(); i++){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>   (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // 核函数？干嘛的？
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back( edge );
    }
    // 第二帧
    for( size_t i = 0; i < pts2.size(); i++ ){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2)) );
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(pts[i].x, pts[i].y) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back( edge );
    }

    cout << "开始优化" << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout << "优化完毕" << endl;

    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> ( optimizer.vertex(1) );




    return 0;
}