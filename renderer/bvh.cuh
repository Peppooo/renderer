#include "scene.cuh"

class boundingBox {
public:
	vec3 edge,size;
	boundingBox() {};
	boundingBox(vec3 Edge,vec3 Size):edge(Edge),size(Size) {};
	__device__ bool intersect(vec3 O,vec3 D) const {
		vec3 invDir = 1 / D;
		vec3 t0s = (edge - O) * invDir;
		vec3 t1s = (edge + size - O) * invDir;

		vec3 tmin3 = v_min(t0s,t1s);
		vec3 tmax3 = v_max(t0s,t1s);

		float tmin = max(max(tmin3.x,tmin3.y),tmin3.z);
		float tmax = min(min(tmax3.x,tmax3.y),tmax3.z);

		// Intersection occurs if tmax >= max(tmin, 0.0)
		return tmax >= max(tmin,0.0f);
	}
};


__host__ void build_bounding_box_from_array(boundingBox& box,const object* scene,const int sceneSize) {
	vec3 min_vert,max_vert;
	for(int i = 0; i < sceneSize; i++) {
		vec3 a = scene[i].a; vec3 b = scene[i].b; vec3 c = scene[i].c;
		min_vert.x = min(min_vert.x,min({a.x,b.x,c.x}));
		min_vert.y = min(min_vert.y,min({a.y,b.y,b.y}));
		min_vert.z = min(min_vert.z,min({a.z,b.z,c.z}));
		max_vert.x = max(min_vert.x,max({a.x,b.x,c.x}));
		max_vert.y = max(max_vert.y,max({a.y,b.y,c.y}));
		max_vert.z = max(max_vert.z,max({a.z,b.z,c.z}));
	}
	box = boundingBox(min_vert,max_vert - min_vert);
}

__host__ void build_bounding_box(boundingBox& box,const Scene* scene) {
	vec3 min_vert,max_vert;
	for(int i = 0; i < scene->sceneSize; i++) {
		vec3 a = scene->a[i]; vec3 b = scene->b[i]; vec3 c = scene->c[i];
		min_vert.x = min(min_vert.x,min({a.x,b.x,c.x}));
		min_vert.y = min(min_vert.y,min({a.y,b.y,b.y}));
		min_vert.z = min(min_vert.z,min({a.z,b.z,c.z}));
		max_vert.x = max(min_vert.x,max({a.x,b.x,c.x}));
		max_vert.y = max(max_vert.y,max({a.y,b.y,c.y}));
		max_vert.z = max(max_vert.z,max({a.z,b.z,c.z}));
	}
	box = boundingBox(min_vert,max_vert - min_vert);
}