// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
module.exports = async function createConfigAsync() {
  return{
  title: 'BUAAer-xing Blog',
  tagline: 'Dinosaurs are cool',
  favicon: 'img/icon.png',

  // Set the production url of your site here
  url: 'https://buaaer-xing.github.io/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'BUAAer-xing', // Usually your GitHub org/user name.
  projectName: 'BUAAer-xing.github.io', // Usually your repo name.
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          remarkPlugins: [(await import('remark-math')).default],
          rehypePlugins: [(await import('rehype-katex')).default],
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          blogSidebarTitle: '记录一些零碎的知识',
          blogSidebarCount: 'ALL',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          remarkPlugins: [(await import('remark-math')).default],
          rehypePlugins: [(await import('rehype-katex')).default],
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
  
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Home',
        logo: {
          alt: 'My Site Logo',
          src: 'img/icon.png',
        },
        items: [
          {
            type: 'doc',
            docId: 'paper_notes/paper-notes-intro',
            position: 'left',
            label: '论文笔记',
          },
          {
            type: 'doc',
            docId: 'week_report/week_report_intro',
            position: 'left',
            label: '周报汇总',
          },
          {
            type: 'doc',
            docId: 'blogs/blogs-intro',
            position: 'left',
            label: '个人博客',
          },
          {
            to: '/blog', 
            label: '相关内容', 
            position: 'left'
          },
          {
            to: '/resume',
            label: '个人简历',
            position: 'left'
          },
          {
            href: 'https://github.com/facebook/docusaurus',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: '论文笔记',
                to: 'docs/paper-notes-intro',
              },
              {
                label: '个人博客',
                to: 'docs/blogs-intro',
              },
              {
                label: '相关内容',
                to: 'blog',
              },
              {
                label: '个人简历',
                to: 'resume',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Telegram',
                href: 'https://t.me/cx_cst',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'CSDN',
                to: 'https://blog.csdn.net/qq_45575167',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/NEUQer-xing',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} BUAAer-xing, 此网站使用 Docusaurus 进行构建✨`,
      },
      docs: {
        sidebar: {
          hideable: true
        }
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      algolia: {
        // The application ID provided by Algolia
        appId: 'YOUR_APP_ID',
  
        // Public API key: it is safe to commit it
        apiKey: 'YOUR_SEARCH_API_KEY',
  
        indexName: 'YOUR_INDEX_NAME',
  
        // Optional: see doc section below
        contextualSearch: true,
  
        // Optional: Specify domains where the navigation should occur through window.location instead on history.push. Useful when our Algolia config crawls multiple documentation sites and we want to navigate with window.location.href to them.
        externalUrlRegex: 'external\\.com|domain\\.com',
  
        // Optional: Algolia search parameters
        searchParameters: {},
  
        // Optional: path for search page that enabled by default (`false` to disable it)
        searchPagePath: 'search',
  
        //... other Algolia params
      },
    }),
  };
};