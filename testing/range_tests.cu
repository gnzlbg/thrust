#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/range.h>

struct reduced : thrust::unary_function<thrust::tuple<int, int, int>, int> {
  int operator()(thrust::tuple<int, int, int> i) {
    return thrust::get<0>(i) + thrust::get<1>(i) + thrust::get<2>(i);
  }
};

template <class Vector>
void TestGeneralRange(void) {
  typedef typename Vector::value_type T;

  typedef thrust::negate<T> UnaryFunction;
  typedef typename Vector::iterator Iterator;

  Vector input1(4);
  thrust::sequence(input1, 1);
  // input1 = {1, 2, 3, 4};

  Vector input2(4);
  thrust::sequence(input2, 2);
  // input2 = {2, 3, 4, 5};

  Vector input3(4);
  thrust::sequence(input3, 3);
  // input3 = {3, 4, 5, 6};

  Vector output(4);

  thrus::copy(zip(input1, input2, input3) | thrust::transformed(reduced()),
              output);
  // output = { sum(1,2,3), sum(2,3,4), sum(3,4,5), sum(4,5,6) = {6, 9, 12, 15};

  ASSERT_EQUAL(output[0], 6);
  ASSERT_EQUAL(output[1], 9);
  ASSERT_EQUAL(output[2], 12);
  ASSERT_EQUAL(output[3], 15);

  // test reduce = 6 + 9 + 12 + 15 = 42
  ASSERT_EQUAL(thrust::reduce(output), 42);

  // test transform_reduce
  ASSERT_EQUAL(thrust::transform_reduce(zip(input1, input2, input3), reduced()),
               42);

  // test counting_range
  thrust::copy(thrust::make_counting_range(0, 4), output);
  ASSERT_EQUAL(output[0], 0);
  ASSERT_EQUAL(output[1], 1);
  ASSERT_EQUAL(output[2], 2);
  ASSERT_EQUAL(output[3], 3);
}
DECLARE_VECTOR_UNITTEST(TestGeneralRange);
