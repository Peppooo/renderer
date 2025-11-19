#pragma once
#include "algebra.cuh"

constexpr int w = 1024,h = 1024;
constexpr float fov = M_PI / 1.5;
constexpr float move_speed = 0.1;
constexpr float mouse_sens = 0.001;
float foc_len = w / (2 * tanf(fov / 2));
bool hq = false;

vec3 origin = {0,0,0};
bool move_light = false;
int current_light_index = 0;
float yaw = 0,pitch = 0,roll = 0;

int reflections = 5;
int ssaa = 4;

int reflected_rays = 8;