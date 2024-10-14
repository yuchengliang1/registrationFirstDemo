cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(MultiResImageRegistration3)

find_package(VTK COMPONENTS 
  CommonColor
  CommonCore
  CommonDataModel
  IOImage
  ImagingCore
  InteractionStyle
  InteractionWidgets
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
)

find_package(ITK REQUIRED)

include(${ITK_USE_FILE})

# set(ITK_REQUIRED_MODULES ITKImageRegistration ITKIO)


if (NOT VTK_FOUND)
  message(FATAL_ERROR "MultiResImageRegistration3: Unable to find the VTK build folder.")
endif()

# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")
add_executable(MultiResImageRegistration3 MACOSX_BUNDLE MultiResImageRegistration3.cxx )
  target_link_libraries(MultiResImageRegistration3 PRIVATE ${VTK_LIBRARIES} ${ITK_LIBRARIES}
)
# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS MultiResImageRegistration3
  MODULES ${VTK_LIBRARIES}
)