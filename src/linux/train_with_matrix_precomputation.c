char m_model_file_name[1024];

void setup_pkm2(struct svm_problem *p_km)
{

    int i;

    p_km->l = prob.l;
    p_km->x = Malloc(struct svm_node,p_km->l);
    p_km->y = Malloc(double,p_km->l);

    for(i=0;i<prob.l;i++)
    {
        (p_km->x+i)->values = Malloc(double,prob.l+1);
        (p_km->x+i)->dim = prob.l+1;
    }

    for( i=0; i<prob.l; i++) p_km->y[i] = prob.y[i];
}

void free_pkm2(struct svm_problem *p_km)
{

    int i;

    for(i=0;i<prob.l;i++)
        free( (p_km->x+i)->values);

    free( p_km->x );
    free( p_km->y );

}

void do_training(struct svm_problem * p_km)
{

    printf("Running do_training...\n");
    model = svm_train(p_km,&param);
    printf("svm_train complete.\n");

    if(svm_save_model(m_model_file_name,model))
    {
        fprintf(stderr, "can't save model to file %s\n", m_model_file_name);
        exit(1);
    } else {
        printf("Saving successful.\n");
    }
    svm_free_and_destroy_model(&model);

}

void run_pair2(struct svm_problem * p_km)
{

    cal_km( p_km);
    printf("Computing kernel matrix completed.\n");

    param.kernel_type = PRECOMPUTED;

    do_training(p_km);
    printf("Training complete.\n");

}

void train_with_KM_precalculated(const char *model_file_name)
{

    strcpy(m_model_file_name, model_file_name);
    printf("Training with precomputed kernel matrix...\n");

    struct svm_problem p_km;

    setup_pkm2(&p_km );
    printf("Setup complete, running kernel computation and training...\n");

    run_pair2( &p_km);

    free_pkm2(&p_km);
    printf("Done freeing memory for kernel matrix.\n");
}
