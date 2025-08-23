import classnames from "classnames";
import type { Component } from "solid-js";
import styles from "./MadeBy.module.scss";

interface Props {
  class?: string;
}

export const MadeBy: Component<Props> = (props) => {
  return (
    <p class={classnames(styles.madeBy, props.class)}>
    </p>
  );
};
