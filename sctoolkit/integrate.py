import scanpy as sc
import numpy as np

from .disvae.disvae.singlecell import fit_single_cell

def integrate(adata, categorical_vars, experiment_name='tcbetavae',beta=2., count_layer='counts', n_hvg=5000, cuda=False, latent_dim=64, epochs=150, output_activation='softplus'):

    ad = adata.copy()
    ad.X = ad.layers[count_layer].copy()
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, subset=True, flavor='seurat_v3')
    sc.pp.normalize_total(ad, target_sum=10000)
    sc.pp.log1p(ad)

    ad, model, trainer, train_loader = fit_single_cell(ad.copy(), experiment_name, btcvae_B=beta, cuda=cuda,
                                                       categorical_vars=categorical_vars,
                                                       latent_dim=latent_dim, epochs=epochs, output_activation=output_activation)

    assert np.all(adata.obs_names == ad.obs_names)

    adata.obsm['X_vae_samples'] = ad.obsm['X_vae_samples']
    adata.obsm['X_vae_mean'] = ad.obsm['X_vae_mean']
    adata.obsm['X_vae_var'] = ad.obsm['X_vae_var']

    return adata, model
