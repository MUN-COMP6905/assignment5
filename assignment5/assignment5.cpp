#include <iostream>
#include <vector>
#include <fstream>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

const int width = 1024;
const int height = 1024;
const int maxIterations = 128;
const float minX = -1.5f;
const float maxX = 5.5f;
const float minY = -1.2f;
const float maxY = 1.2f;

struct Complex {
    float x, y;
};

std::vector<int> generateImage() {
    std::vector<int> image(width * height);

    std::vector<Complex> points(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            points[y * width + x].x = minX + (maxX - minX) * x / (width - 1);
            points[y * width + x].y = minY + (maxY - minY) * y / (height - 1);
        }
    }


    std::vector<int> iterations(width * height);

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);
    const char* source = R"CLC(
        typedef struct {
            float x, y;
        } Complex;

        __kernel void phoenix(__global int* iterations, __global Complex* points) {
            int gid = get_global_id(0);
            Complex z = points[gid];

            for (int i = 0; i < 128; ++i) {
                Complex temp;
                temp.x = z.x * z.x + z.y * (-0.5f) + 0.56667f;
                temp.y = z.x;
                z = temp;

                if ((z.x * z.x + z.y * z.y) > 4.0f) {
                    iterations[gid] = i;
                    return;
                }
            }

            iterations[gid] = 128;
        }
    )CLC";

    program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "phoenix", nullptr);

    cl_mem bufferPoints = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(Complex) * points.size(), points.data(), nullptr);
    cl_mem bufferIterations = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(int) * iterations.size(), nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferIterations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferPoints);

    size_t globalSize = width * height;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bufferIterations, CL_TRUE, 0, sizeof(int) * iterations.size(), iterations.data(), 0, nullptr, nullptr);

    clReleaseMemObject(bufferPoints);
    clReleaseMemObject(bufferIterations);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);



    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float value = static_cast<float>(iterations[y * width + x]) / maxIterations;
            image[y * width + x] = static_cast<int>(255 * value);
        }
    }

    return image;
}
void saveBMP(const std::string& filename, const std::vector<int>& image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error" << std::endl;
        return;
    }

    int imageSize = width * height;
    int fileSize = 54 + 3 * imageSize;
    char header[54] = {
        'B', 'M', fileSize & 0xFF, (fileSize >> 8) & 0xFF, (fileSize >> 16) & 0xFF, fileSize >> 24,
        0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, width & 0xFF, (width >> 8) & 0xFF, 0, 0,
        height & 0xFF, (height >> 8) & 0xFF, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    file.write(header, 54);
    for (int i = 0; i < imageSize; ++i) {
        char pixel[3] = { static_cast<char>(image[i]), static_cast<char>(image[i]), static_cast<char>(image[i]) };
        file.write(pixel, 3);
    }

    file.close();
}

int main() {
    std::vector<int> image = generateImage();
    saveBMP("phoenix_curve.bmp", image);

    return 0;
}
