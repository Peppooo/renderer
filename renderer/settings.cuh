#pragma once
#include "objects.cuh"

constexpr int w = 1024,h = 1024;
constexpr float fov = M_PI / 1.4f;
constexpr float move_speed = 0.1f;
constexpr float mouse_sens = 0.001f;
float foc_len = w / (2 * tanf(fov / 2.0f));
bool hq = false;

vec3 origin = {0.0f,0.0f,-2.0f};
bool move_light = false;
int current_light_index = 0;
float yaw = 0.0f,pitch = 0.0f,roll = 0.0f;

int reflections = 5;
int ssaa = 1;

__constant__ int reflected_rays = 256;