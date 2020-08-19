/**
 * @file methods/ann/loss_functions/multilabel_margin_loss.hpp
 * @author R Aravind
 *
 * Definition of the Multi-Label Margin Loss function.
 *
 * It is a criterion that optimizes a multi-class multi-classification hinge  
 * loss between input x and output y, both having same shape, with the target 
 * being a one hot tensor with 1’s at each target class index and 0’s at every 
 * other index.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_ANN_LOSS_FUNCTION_MULTILABEL_MARGIN_LOSS_HPP
#define MLPACK_ANN_LOSS_FUNCTION_MULTILABEL_MARGIN_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. **/ {

template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class MultiLabelMarginLoss
{
 public:
  MultiLabelMarginLoss(const bool reduction = true);
 
  template<typename InputType, typename TargetType>
  typename InputType::elem_type Forward(const InputType& input,
                                       const TargetType& target);

  template<typename InputType, typename TargetType>
  void Backward(const InputType& input,
                const TargetType& target,
                OutputType& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the type of reduction used.
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The boolean value that tells if reduction is sum or mean.
  bool reduction;

} // class MultiLabelMarginLoss

} // namespace ann
} // namespace mlpack

#endif
