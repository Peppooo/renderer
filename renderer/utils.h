#pragma once
#include <vector>

using namespace std;

struct vec3 {
	double x,y,z;
	vec3 operator*(double scalar) {
		return {x * scalar,y * scalar,z * scalar};
	}
	vec3 operator/(double scalar) {
		return {x / scalar,y / scalar,z / scalar};
	}
	vec3 operator+(vec3 a) {
		return {x + a.x,y + a.y,z + a.z};
	}
	vec3 operator-(vec3 a) {
		return {x - a.x,y - a.y,z - a.z};
	}
	vec3 operator-() {
		return {-x,-y,-z};
	}
	vec3 operator*(vector<vec3> matrix) {
		if(matrix.size() != 3) {
			throw "matrix with wrong size";
		}
		return matrix[0] * x + matrix[1] * y + matrix[2] * z;
	}

	double len() {
		return sqrt(x * x + y * y + z * z);
	}
	vec3 norm() {
		return *this / len();
	}
};

vec3 cross(vec3 a,vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

double dot(vec3 a,vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

vector<vec3> rotation(double yaw,double pitch,double roll) {
	return {
		{cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll),-cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll),sin(yaw) * cos(pitch)},
		{sin(roll) * cos(pitch),cos(roll) * cos(pitch),-sin(pitch)},
		{-sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll),cos(yaw) * cos(pitch)}
	};
}

vector<vec3> rotationYaw(double theta) {
	return {
		{cos(theta),0,sin(theta)},{0,1,0},{-sin(theta),0,cos(theta)} // the rotation matrix is different since i use Y as height X at the right and Z as far
	};
}

class trig {
public:
	vec3 a,b,c;
	vec3 color;
	static vector<trig*> triangles;
	trig(vec3 A,vec3 B,vec3 C,vec3 Color):a(A),b(B),c(C),color(Color) {
		triangles.push_back(this);
	}
	bool intersect(vec3 O,vec3 D,vec3& p,vec3& N) {
		N = (cross(b - a,c - a));
		double t = dot(N,a - O) / dot(N,D);
		if(t < 0) return false;
		p = O + D * t;
		vec3 v0 = c - a;
		vec3 v1 = b - a;
		vec3 v2 = p - a;
		double d00 = dot(v0,v0);
		double d01 = dot(v0,v1);
		double d11 = dot(v1,v1);
		double d20 = dot(v2,v0);
		double d21 = dot(v2,v1);
		double denom = d00 * d11 - d01 * d01;
		double v = (d11 * d20 - d01 * d21) / denom,u = (d00 * d21 - d01 * d20) / denom;
		return (u >= 0) && (v >= 0) && (u + v <= 1);
	}
};

vector<trig*> trig::triangles;

trig* castRay(vec3 O,vec3 D,vec3& p,vec3& n) {
	trig* closest = nullptr;
	double closest_dist = INFINITY;
	for(trig* t : trig::triangles) {
		vec3 temp_p,temp_n;
		if(t->intersect(O,D,temp_p,temp_n)) {
			double dist = (temp_p - O).len();
			if(dist < closest_dist) {
				closest_dist = dist;
				closest = t;
				p = temp_p;
				n = temp_n;
			}
		}
	}
	return closest;
}

SDL_Color compute_ray(vec3 O,vec3 D,vec3 light,int reflections = 2) {
	vec3 color; int done_reflections = 0;
	for(int i = 0; i < reflections; i++) {
		vec3 p,surf_norm; // p is the intersection location
		trig* triangle = castRay(O,D,p,surf_norm);
		if(triangle != nullptr) {
			if(dot(surf_norm,D) > 0) {
				surf_norm = -surf_norm; // adjust the surface normal so its in the opposite direction from the ray direction
			}

			double scalar = dot((light - p).norm(),surf_norm.norm()); // how much light is in a point is calculated by the dot product of the surface normal and the direction of the surface point to the light point
			
			if(scalar > 0) { // if the point is not facing the light it will not be drawn
				if(done_reflections == 0) {
					color = triangle->color * scalar;
				}
				else {
					color = color + (triangle->color * scalar);
				}
				done_reflections++;
				O = p;
				D = D - surf_norm * 2 * dot(surf_norm,D); // reflection based on surface normal
			}
			else {
				break;
			}
		}
	}
	color = color / done_reflections;

	return {(Uint8)color.x,(Uint8)color.y,(Uint8)color.z,255}; // i use vec3 for colors since it has operator configured and i don't care about alpha channel
}