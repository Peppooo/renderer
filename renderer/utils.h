#pragma once
#include <vector>

using namespace std;

const double epsilon = 1e-7;

bool IS_CENTER_PIXEL = false;

int cycle(int i,int max) {
	if(i == max) {
		return 0;
	}
	return i;
}

struct vec3 {
	double x,y,z;
	vec3 operator*(const double& scalar) const {
		return {x * scalar,y * scalar,z * scalar};
	}
	vec3 operator/(const double& scalar) const {
		return {x / scalar,y / scalar,z / scalar};
	}
	vec3 operator+(const vec3& a) const {
		return {x + a.x,y + a.y,z + a.z};
	}
	vec3 operator-(const vec3& a) const {
		return {x - a.x,y - a.y,z - a.z};
	}
	vec3 operator-() const {
		return {-x,-y,-z};
	}
	vec3 operator*(const vector<vec3>& matrix) const {
		if(matrix.size() != 3) {
			throw "matrix with wrong size";
		}
		return matrix[0] * x + matrix[1] * y + matrix[2] * z;
	}
	void operator+=(const vec3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}
	void operator*= (const double& scalar){
		x *= scalar;
		y *= scalar;
		z *= scalar;
	}
	void operator*=(const vector<vec3>& matrix) {
		if(matrix.size() != 3) {
			throw "matrix with wrong size";
		}
		*this = matrix[0] * x + matrix[1] * y + matrix[2] * z;
	}
	double len() const {
		return sqrt(x * x + y * y + z * z);
	}
	vec3 norm() const {
		return *this / len();
	}
	uint32_t argb() {
		return (255 << 24) | ((Uint8)x << 16) | ((Uint8)y << 8) | (Uint8)z;
	}
};

vec3 cross(const vec3& a,const vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

double dot(const vec3& a,const vec3& b) {
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
		{cos(theta),0,sin(theta)},{0,1,0},{-sin(theta),0,cos(theta)} // the rotation matrix is different since i use Y as up X as right and Z as forward
	};
}

class trig {
public:
	vec3 a,b,c;
	vec3 color;
	bool reflective = false;
	static vector<trig*> triangles;
	trig(vec3 A,vec3 B,vec3 C,vec3 Color,bool Reflective=false):a(A),b(B),c(C),color(Color),reflective(Reflective) {
		triangles.push_back(this);
	}
	bool intersect(const vec3& O,const vec3& D,vec3& p,vec3& N) {
		N = (cross(b - a,c - a)).norm();
		if(dot(N,D) > 0) {
			N = -N; // adjust the surface normal so its in the opposite direction from the ray direction
		}
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
		p = p + N * epsilon;
		return (u >= 0) && (v >= 0) && (u + v <= 1);
	}
};

vector<trig*> trig::triangles;

class cube {
public:
	cube(vec3 edge,double lx,double ly,double lz,vec3 color,bool reflective = false) {
		new trig(edge,edge + vec3{lx,0,0},edge + vec3{0,ly,0},color,reflective);
		new trig(edge + vec3{0,ly,0},edge+vec3{lx,ly,0},edge + vec3{lx,0,0},color,reflective);
		new trig(edge,edge + vec3{0,0,lz},edge + vec3{0,ly,0},color,reflective);
		new trig(edge + vec3{0,0,lz},edge + vec3{0,ly,lz},edge + vec3{0,ly,0},color,reflective);
		new trig(edge + vec3{lx,0,lz},edge + vec3{lx,ly,lz},edge + vec3{lx,0,0},color,reflective);
		new trig(edge + vec3{lx,0,0},edge + vec3{lx,ly,0},edge + vec3{lx,ly,lz},color,reflective);
		new trig(edge + vec3{lx,0,lz},edge+vec3{0,0,lz},edge+ vec3{lx,ly,lz},color,reflective);
		new trig(edge + vec3{lx,ly,lz},edge+vec3{0,ly,lz},edge+ vec3{0,0,lz},color,reflective);
		new trig(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,reflective);
		new trig(edge + vec3{0,0,0},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,reflective);
		new trig(edge + vec3{lx,0,lz},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,reflective);
		new trig(edge + vec3{lx,ly,lz},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,reflective);
	}
};

trig* castRay(const vec3& O,const vec3& D,vec3& p,vec3& n) {
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

double compute_light_scalar(const vec3& p,const vec3& n,const vector<vec3>& lights,bool& visible) { // gets illumination from the brightest of all the lights
	double max_light_scalar = 0;
	visible = false;
	for(int i = 0; i < lights.size(); i++) {
		vec3 pl,nl;
		double scalar = (dot((lights[i] - p).norm(),n.norm())); // how much light is in a point is calculated by the dot product of the surface normal and the direction of the surface point to the light point
		if(IS_CENTER_PIXEL) {

		}
		if(scalar > max_light_scalar && (castRay(p,(lights[i] - p).norm(),pl,nl) == nullptr || (pl-p).len()>=(p-lights[i]).len())) {
			max_light_scalar = scalar;
		}
	}
	if(max_light_scalar < 0) {
		visible = false;
	}
	else {
		visible = true;
	}
	return max_light_scalar;
}

vec3 compute_ray(vec3 O,vec3 D,const vector<vec3>& lights,int reflections = 2) {
	vec3 color = {0,0,0}; int done_reflections = 0;
	vector<double> scalars;
	for(int i = 0; i < reflections; i++) {
		vec3 p,surf_norm = {0,0,0}; // p is the intersection location
		trig* triangle = castRay(O,D,p,surf_norm);
		if(triangle != nullptr) {
			bool visible;
			double scalar = compute_light_scalar(p,surf_norm,lights,visible);
			vec3 pl,nl;
			if(visible) { // if the point is not facing the light it will not be drawn
				color += (triangle->color * scalar);
				done_reflections++;
				if(done_reflections < reflections && triangle->reflective) { // if its not the last reflection or triangle hit not reflective
					O = p;
					D = (D - surf_norm * 2 * dot(surf_norm,D)).norm(); // reflection based on surface normal
				}
			}
			if(!triangle->reflective) {
				break;
			}
		}
		else {
				break;
		}
	}
	return color / done_reflections;
}