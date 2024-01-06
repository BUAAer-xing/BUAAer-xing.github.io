import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <div className="hero" >
      <div className={styles.welcome_intro}>
        <h1 className={styles.hero_title}>
          凡事发生必有
          <span
            style={{ color: 'var(--ifm-color-primary)' }}
          >利于</span>我</h1>
        <h1 className={styles.hero_title}>
          没有失败只有
          <span style={{color:'var(--ifm-color-warning)'}}>经验❗️</span> </h1>
        <p className="hero__subtitle">好记性不如烂笔头，日积月累，水滴石穿～</p>
      </div>
      <div className={styles.welcome_svg}>
        <img src={useBaseUrl("/img/program.svg")} />
      </div>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout>
      <HomepageHeader />
      <main>
        <br/>
        <br/>
        <br/>
      </main>
    </Layout>
  );
}
