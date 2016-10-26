#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossBDKLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_1_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_1_.clear();
  softmax_bottom_vec_1_.push_back(bottom[2]);
  softmax_top_vec_1_.clear();
  softmax_top_vec_1_.push_back(&prob_1_);
  softmax_layer_1_->SetUp(softmax_bottom_vec_1_, softmax_top_vec_1_);

}

template <typename Dtype>
void SoftmaxWithLossBDKLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_1_->Reshape(softmax_bottom_vec_1_, softmax_top_vec_1_);
}

template <typename Dtype>
void SoftmaxWithLossBDKLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  SoftmaxWithLossLayer<Dtype>::Forward_cpu(bottom, top);
  softmax_layer_1_->Forward(softmax_bottom_vec_1_, softmax_top_vec_1_);
}

template <typename Dtype>
void SoftmaxWithLossBDKLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    int count = 0;
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff1 = bottom[2]->mutable_cpu_diff();
    const Dtype* prob_data = this->prob_.cpu_data();
    caffe_copy(this->prob_.count(), prob_data, bottom_diff);
    caffe_copy(this->prob_.count(), prob_data, bottom_diff1);
    if(this->layer_param_.loss_param().down_sgld() == 0){
      caffe_scal(this->prob_.count(), Dtype(0), bottom_diff);
    }else{
    const Dtype* label = bottom[1]->cpu_data();
    int dim = this->prob_.count() / this->outer_num_;
    for (int i = 0; i < this->outer_num_; ++i) {
      for (int j = 0; j < this->inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * this->inner_num_ + j]);
        if (this->has_ignore_label_ && label_value == this->ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(this->softmax_axis_); ++c) {
            bottom_diff[i * dim + c * this->inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * this->inner_num_ + j] -= 1;
          ++count;
        }
      }
    }}
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_cpu_axpby(this->prob_.count(), Dtype(1), this->prob_1_.cpu_data(), Dtype(-1), bottom_diff1);
    if (this->normalize_) {
      if(this->layer_param_.loss_param().down_sgld() == 1){
      caffe_scal(this->prob_.count(), loss_weight / count, bottom_diff);}
      caffe_scal(this->prob_.count(), loss_weight / count, bottom_diff1);
      //caffe_copy(this->prob_.count(), bottom_diff, bottom_diff1);
    } else {
      if(this->layer_param_.loss_param().down_sgld() == 1){
      caffe_scal(this->prob_.count(), loss_weight / this->outer_num_, bottom_diff);}
      caffe_scal(this->prob_.count(), loss_weight / this->outer_num_, bottom_diff1);
      //caffe_copy(this->prob_.count(), bottom_diff, bottom_diff1);
    }
//for(int i = 0; i < 10; i++){LOG(INFO) << bottom_diff1[i];}LOG(INFO) << "size = " << this->prob_.count();
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossBDKLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossBDKLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLossBDK);

}  // namespace caffe

