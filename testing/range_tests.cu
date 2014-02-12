#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/range.h>

struct reduced : thrust::unary_function<thrust::tuple<int, int, int>, int> {
  int operator()(thrust::tuple<int, int, int> i) {
    return thrust::get<0>(i) + thrust::get<1>(i) + thrust::get<2>(i);
  }
};

struct assert_range : thrust::unary_function<int, void> {
  assert_range(int b_, int e_) : b(b_), e(e_) {}
  void operator()(int i) const {
    ASSERT_GEQUAL(i, b);
    ASSERT_LEQUAL(i, e - 1);
  }
  const int b;
  const int e;
};

template <class Vector>
void TestGeneralRange(void) {
  typedef typename Vector::value_type T;

  typedef thrust::negate<T> Negate;
  typedef typename Vector::iterator Iterator;
  //typedef thrust::iterator_range<Iterator> IteratorRange;


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

  thrust::copy(thrust::zip(input1, input2, input3)
               | thrust::transformed(reduced()),
              output);
  // output = { sum(1,2,3), sum(2,3,4), sum(3,4,5), sum(4,5,6) = {6, 9, 12, 15};

  thrust::for_each(output, assert_range(6, 16));  // should pass

  ASSERT_EQUAL(output[0], 6);
  ASSERT_EQUAL(output[1], 9);
  ASSERT_EQUAL(output[2], 12);
  ASSERT_EQUAL(output[3], 15);

  Vector output_pipe(4);
  thrust::zip(input1, input2, input3)
      | thrust::transformed(reduced()) | thrust::copy(output_pipe);

  ASSERT_EQUAL(output_pipe[0], 6);
  ASSERT_EQUAL(output_pipe[1], 9);
  ASSERT_EQUAL(output_pipe[2], 12);
  ASSERT_EQUAL(output_pipe[3], 15);
  thrust::sequence(output_pipe, 1);

  thrust::sort(output, thrust::greater<T>());
  ASSERT_EQUAL(output[0], 15);
  ASSERT_EQUAL(output[1], 12);
  ASSERT_EQUAL(output[2], 9);
  ASSERT_EQUAL(output[3], 6);
  thrust::sort(output);

  thrust::zip(input1, input2, input3)
      | thrust::transformed(reduced())
      | thrust::copy(output_pipe)
      | thrust::sorted(thrust::greater<T>())
      | thrust::transformed(Negate())
      | thrust::copy(output_pipe);
  ASSERT_EQUAL(output_pipe[0], -15);
  ASSERT_EQUAL(output_pipe[1], -12);
  ASSERT_EQUAL(output_pipe[2], -9);
  ASSERT_EQUAL(output_pipe[3], -6);

  thrust::transformed_range<const Vector, Negate> lazy_tr
      = thrust::transform(output, Negate());

  Vector copy1(4);
  thrust::copy(lazy_tr, copy1);

  ASSERT_EQUAL(copy1[0], -6);
  ASSERT_EQUAL(copy1[1], -9);
  ASSERT_EQUAL(copy1[2], -12);
  ASSERT_EQUAL(copy1[3], -15);

  Vector output2(4);
  thrust::transform(output, output2, Negate());
  ASSERT_EQUAL(output2[0], -6);
  ASSERT_EQUAL(output2[1], -9);
  ASSERT_EQUAL(output2[2], -12);
  ASSERT_EQUAL(output2[3], -15);

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
