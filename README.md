# Microstructure reconstruction

## Github Usage

Here are a few links to use github, with the terminal:

* <https://gitimmersion.com/index.html>
* <http://up1.github.io/git-guide/index.html>

But you may also use github with the desktop application:

* <https://www.softwaretestinghelp.com/github-desktop-tutorial/>

You may also need to set up a SSH connection to github. Basically, a SSH is a secure protocol that allows you to connect to private repository:

* <https://jdblischak.github.io/2014-09-18-chicago/novice/git/05-sshkeys.html>

## Tree of directories

The script are currently working with this directories / naming conventions:

```bash
.
├── MATLAB
│   ├── README.md
│   ├── grain.m
│   ├── import_stl.mlx
│   ├── rev.m
├── REV1_600
│   ├── REV1_6003D_model
│   │   ├── Spec-1.STL
│   │   ├── Spec-2.STL
│   │   └── ...
│   ├── REV1_600Slices
│   │   ├── 1pics
│   │   │   ├── Spec-1_Imgs
│   │   │   │   ├── *.png
│   │   │   │   ├── *.png
│   │   │   │   └── ...
│   │   │   ├── Spec-2_Imgs
│   │   │   │   ├── *.png
│   │   │   │   ├── *.png
│   │   │   │   └── ...
│   │   │   ├── ...
│   │   ├── 3pics
│   │   └── ...
│   └── fabrics.txt
├── README.md <- YOU ARE CURRENTLY HERE
└── ...
```

## Contents

For now, the repository contains the computation of fabrics with MATLAB
