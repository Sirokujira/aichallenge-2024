cmake_minimum_required(VERSION 3.8)
project(simple_pure_pursuit)

# AutowareのCMakeモジュールを使用
find_package(autoware_cmake REQUIRED)
autoware_package()

#find_package(ipopt REQUIRED)
find_package(PkgConfig REQUIRED)

# カスタム pkgconfig フォルダのパスを設定(登録済み)
#set(ENV{PKG_CONFIG_PATH} "/aichallenge/workspace/install/casadi/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
# カスタム CMake モジュールディレクトリを設定
set(CMAKE_MODULE_PATH "/aichallenge/workspace/install/casadi/lib/cmake" ${CMAKE_MODULE_PATH})
#set(CMAKE_MODULE_PATH "/aichallenge/workspace/install/casadi/lib/cmake/casadi" ${CMAKE_MODULE_PATH})

message(STATUS "*********************************")
message(STATUS "$ENV{PKG_CONFIG_PATH}")
message(STATUS ${CMAKE_MODULE_PATH})
# 必要な依存関係を見つける
find_package(casadi REQUIRED)
pkg_check_modules(ipopt REQUIRED ipopt)
message(STATUS "*********************************")

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${ipopt_INCLUDE_DIRS})
link_directories(${ipopt_LIBRARY_DIRS})

message(STATUS "*********************************")
message(STATUS ${ipopt_INCLUDE_DIRS})
message(STATUS ${ipopt_LIBRARY_DIRS})
message(STATUS ${ipopt_LIBRARIES})
#x
message(STATUS ${casadi_INCLUDE_DIRS})
message(STATUS ${casadi_LIBRARY_DIRS})
message(STATUS ${casadi_LIBRARIES})
message(STATUS "*********************************")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies(REQUIRED
  ${${PROJECT_NAME}_BUILD_DEPENDS}
  ${${PROJECT_NAME}_BUILDTOOL_DEPENDS}
)

# 実行可能ファイルを追加
ament_auto_add_executable(${PROJECT_NAME} 
  src/simple_pure_pursuit.cpp
)

# CasADiライブラリをリンク
#target_link_libraries(${PROJECT_NAME} 
#  ${CASADI_LIBRARIES}
#)
#link_directories("/aichallenge/workspace/install/casadi/lib")
target_link_libraries(${PROJECT_NAME} casadi ${ipopt_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} /aichallenge/workspace/install/casadi/lib/libcasadi_nlpsol_ipopt.so)
#ok
#target_link_libraries(${PROJECT_NAME} casadi ${ipopt_LIBRARIES} /aichallenge/workspace/install/casadi/lib/libcasadi_nlpsol_ipopt.so)
#x
#target_link_libraries(${PROJECT_NAME} casadi::casadi)

# CasADiのインクルードディレクトリを指定(↑の対応ですべて対応)
#target_include_directories(${PROJECT_NAME} PRIVATE
#  ${CASADI_INCLUDE_DIRS}
#)

# パッケージ化
ament_auto_package()

