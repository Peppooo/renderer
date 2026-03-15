#include "scene.cuh"
#include <fstream>
#include <vector>
#include <string>

__host__ vector<string> split_string(const string& s,const char div) {
    size_t start = 0;
    size_t end = s.find(div);
    vector<string> split;
    while(end != string::npos) {
        split.push_back(s.substr(start,end - start));
        start = end + 1;
        end = s.find(div,start);
    }
    split.push_back(s.substr(start));
    return split;
}

__host__ void load_obj_in_host_array_scene(const char* filename,const vec3& position,const vec3& scaling,const material& mat,texture* tex,object* scene,size_t& sceneSize) {
    printf("loading obj... ");
    vector<vec3> verticies;
    vector<vec2> texture_verticies;
    //vector<vec3> normals;
    verticies.reserve(10000000);
    texture_verticies.reserve(10000000);
    //normals.reserve(10000000);
    ifstream file(filename);// here i will use fstream since its easier to use for reading line by line obj files

    if(!file) printf("Error opening file");
    vec3 obj_min = vec3::One * INFINITY; vec3 obj_max = vec3::One * -INFINITY;
    string line;
    while(getline(file,line)) {

        vector<string> split_space = split_string(line,' ');
        
        if(split_space[0] == "v") {
            verticies.push_back(vec3{stof(split_space[1]),stof(split_space[2]),stof(split_space[3])}*scaling);
            obj_min = v_min(obj_min,verticies.back());
            obj_max = v_max(obj_max,verticies.back());
        }
        else if(split_space[0] == "vn") {
            //normals.push_back({stof(split_space[1]),stof(split_space[2]),stof(split_space[3])});
        }
        else if(split_space[0] == "vt") { // texture vertice
            texture_verticies.push_back({stof(split_space[1]),stof(split_space[2])});
        }
        else if(split_space[0] == "f") {
            vector<vec3> vertici_trig;
            vector<vec2> tex_vertici_trig;
            vec3 normal;
            for(int i = 1; i < 4; i++) {
                vector<string> split_indexs = split_string(split_space[i],'/');
                //if(i==1)  normal = normals[stoi(split_indexs[2])-1];
                vertici_trig.push_back(position+(verticies[stoi(split_indexs[0])-1])-obj_min);
                if(!split_indexs[1].empty()) {
                    tex_vertici_trig.push_back(texture_verticies[stoi(split_indexs[1]) - 1]);
                }
            }

            object(vertici_trig[0],vertici_trig[1],vertici_trig[2],scene,sceneSize,mat,tex,false);
            if(!tex_vertici_trig.empty()) {
                scene[sceneSize - 1].t_a = tex_vertici_trig[0];
                scene[sceneSize - 1].t_b = tex_vertici_trig[1];
                scene[sceneSize - 1].t_c = tex_vertici_trig[2];
            }
        }
    }

    file.close(); 
    printf("done\n");
}