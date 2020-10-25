import scanpy as sc

from .disvae.disvae.singlecell import fit_single_cell

def integrate(adata, categorical_vars, experiment_name='tcbetavae',beta=2., count_layer='counts', n_hvg=5000, cuda=False, latent_dim=64, epochs=150, output_activation='softplus'):

    adata.X = adata.layers[count_layer].copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True, flavor='seurat_v3')
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)

    ad, model, trainer, train_loader = fit_single_cell(adata.copy(), experiment_name, btcvae_B=beta, cuda=cuda,
                                                       categorical_vars=categorical_vars,
                                                       latent_dim=latent_dim, epochs=epochs, output_activation=output_activation)

    return ad, model
