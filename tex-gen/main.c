#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>

#define HELP(msg) {printf("%s\nformat:\n %s \n\t-a \"albedo path\" \n\t-n \"normal map path\" \n\t-o \"output path\"\n",msg,argv[0]);return 1;}


int main(int argc,char** argv) {
    if(argc <= 1) {
        HELP("use at least albedo or normal");
    }

    char* alb_path = NULL;
    char* nm_path = NULL;
    char* out_path = ".\\out.bin";

    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"-a")) {
            i++;
            if(i >= argc) HELP("no argument for -a");
            alb_path = argv[i];
        }
        else if(!strcmp(argv[i],"-n")) {
            i++;
            if(i >= argc) HELP("no argument for -n");
            nm_path = argv[i];
        }
        else if(!strcmp(argv[i],"-o")) {
            i++;
            if(i >= argc) HELP("no argument for -o");
            out_path = argv[i];
        }
        else {
            HELP("parameter not recognized");
        }
    }

    FILE* file = fopen(out_path,"wb");

    printf("generating texture (albedo path: %s,normal map path: %s,output path: %s) ... ",alb_path?alb_path:"none",nm_path?nm_path:"none",out_path);

    unsigned char info = (alb_path?(1<<0):(0)) | (nm_path?(1 << 1):(0));
    fwrite(&info,1,1,file);

    if(alb_path) {
        int alb_w,alb_h;
        unsigned char* data = stbi_load(alb_path,&alb_w,&alb_h,NULL,3);
        if(!data) {
            printf("aborted\n error reading albedo file\n");
            return 1;
        }
        fwrite(&alb_w,sizeof(int),1,file);
        fwrite(&alb_h,sizeof(int),1,file);
        fwrite(data,sizeof(unsigned char),alb_w*alb_h*3,file);

    }
    if(nm_path) { // albedo + normal map
        int nm_w,nm_h;
        unsigned char* data = stbi_load(nm_path,&nm_w,&nm_h,NULL,3);
        if(!data) {
            printf("aborted\n error reading albedo file\n");
            return 1;
        }
        fwrite(&nm_w,sizeof(int),1,file);
        fwrite(&nm_h,sizeof(int),1,file);
        fwrite(data,1,nm_w*nm_h*3,file);

    }

    fclose(file);
    printf("done\n");

    return 0;
}