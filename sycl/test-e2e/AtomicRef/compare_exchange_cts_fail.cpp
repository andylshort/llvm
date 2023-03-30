// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
#include <sycl/sycl.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/atomic_ref.hpp>

// This is a stripped down version of 
// https://github.com/KhronosGroup/SYCL-CTS/blob/SYCL-2020/tests/atomic_ref/atomic_ref_compare_exchange_test_core.cpp
// to reproduce an issue

using namespace sycl;

using atomic_ref_type = atomic_ref<int, memory_order::relaxed,
  memory_scope::work_item, access::address_space::generic_space>;

int main() {

  // Test criteria
  memory_order success_order = memory_order::relaxed;
  memory_order failure_order = memory_order::relaxed;
  memory_scope scope = memory_scope::work_item;
  constexpr int Expected = 42;
  constexpr int Desired = 1;
  constexpr bool check_with_equal_values = true;
  constexpr int result_buf_size = []{
    if constexpr (check_with_equal_values)
      return 4;
    else
      return 6;
  }();
  

  queue Q;

  std::array<bool, result_buf_size> result{};

  {
    sycl::buffer result_buf(result.data(), sycl::range(result_buf_size));

    // Testing for local scope/accessor but this also happens in global scope, also
    Q
      .submit([&](sycl::handler& cgh) {
        auto res_accessor = result_buf.template get_access<sycl::access_mode::write>(cgh);

        sycl::local_accessor<int, 1> loc_acc(sycl::range<1>(1), cgh);

        cgh.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) {

          loc_acc[0] = Expected;

          atomic_ref_type a_r(loc_acc[0]);

          if (check_with_equal_values) {
            int expected = Expected;
            int desired = Desired;

            bool success = a_r.compare_exchange_weak(expected, desired, success_order,
                                         failure_order, scope);

            int check_number = 1;

            // Check
            // Equivalent to check_cmpr_exch(res_accessor, success, loc_acc[0], desired, check_number++);
            bool expd_res_of_comp_exch_op;
            if (loc_acc[0] == desired)
              expd_res_of_comp_exch_op = true;
            else
              expd_res_of_comp_exch_op = false;
            // set res[0] to the result
            res_accessor[(check_number) - 1] = success == expd_res_of_comp_exch_op;
            check_number += 1;


            loc_acc[0] = expected;

            success = a_r.compare_exchange_weak(expected, desired, success_order,
                                         scope);

            // Check
            // Equivalent to check_cmpr_exch(res_accessor, success, loc_acc[0], desired, check_number);
            if (loc_acc[0] == desired)
              expd_res_of_comp_exch_op = true;
            else
              expd_res_of_comp_exch_op = false;
            // set res[1] to the result of the test
            res_accessor[(check_number) - 1] = success == expd_res_of_comp_exch_op;

          } else {

            int expected = Desired;
            int desired = Expected;

            auto success = a_r.compare_exchange_strong(expected, desired, success_order,
              failure_order, scope);

            res_accessor[0] = success == false;
            res_accessor[1] = expected == loc_acc[0];

            expected = Desired;

            auto another_success = a_r.compare_exchange_strong(expected, desired,
              success_order, scope);

            res_accessor[2] = another_success == false;
            res_accessor[3] = expected == loc_acc[0];

            res_accessor[4] = std::is_same_v<decltype(success), bool>;
            res_accessor[5] = std::is_same_v<decltype(another_success), bool>;
          }
        });
      });
  }

  // Print out result array for debugging
  for (auto& res : result) {
    std::cout << (res ? "True" : "False") << ", ";
  }
  std::cout << std::endl;

  if (check_with_equal_values) {
    assert(result[1] && "compare_exchange_overloaded call failed");
  } else {
    assert(result[1] && "Error, \"expected\" argument value is not updated after compare_exchange call with uneq values");
    
    assert(result[3] && "Error, \"expected\" argument value is not updated after compare_exchange_overloaded call with uneq values");
  }
}