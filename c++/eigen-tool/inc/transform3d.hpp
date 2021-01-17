
#include <Eigen/Core>

    Eigen::MatrixXf compute_similarity_transform(Eigen::MatrixXf src, Eigen::MatrixXf dst);

    void P2sRt(Eigen::MatrixXf P, float& s, Eigen::MatrixXf& R, Eigen::VectorXf& t);

