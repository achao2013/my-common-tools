#include <iostream>
#include "transform3d.hpp"

    Eigen::MatrixXf compute_similarity_transform(Eigen::MatrixXf src, Eigen::MatrixXf dst)
    {
        Eigen::VectorXf t0 = -src.colwise().mean();
        Eigen::VectorXf t1 = -dst.colwise().mean();
        Eigen::VectorXf t_final = t1-t0;

        src.rowwise()+=t0.transpose();
        dst.rowwise()+=t1.transpose();
        Eigen::MatrixXf covariance_matrix = src.transpose()*dst;
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(covariance_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::MatrixXf V = svd.matrixV();
        Eigen::MatrixXf R = U*V.transpose();
        if (R.determinant() < 0) 
        {    R.col(2) *= -1;  }

        Eigen::VectorXf sn0 = src.rowwise().squaredNorm();
        float rms_d0=std::sqrt(sn0.mean());
        Eigen::VectorXf sn1 = dst.rowwise().squaredNorm();
        float rms_d1=std::sqrt(sn1.mean());
        float s = rms_d0/rms_d1;
        Eigen::MatrixXf P(3,4);
        P<<s*R,t_final;
        return P;
    }
    void P2sRt(Eigen::MatrixXf P, float& s, Eigen::MatrixXf& R, Eigen::VectorXf& t)
    {
        t = P.block(0,3,3,1);
        Eigen::VectorXf R1 = P.block(0,0,1,3).transpose();
        Eigen::VectorXf R2 = P.block(1,0,1,3).transpose();
        s = (R1.norm()+R2.norm())/2;
        Eigen::Vector3f r1 = R1/R1.norm();
        Eigen::Vector3f r2 = R2/R2.norm();
        Eigen::Vector3f r3 = r1.cross(r2);
        R <<r1.transpose(),
            r2.transpose(),
            r3.transpose();

    }

